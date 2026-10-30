"""SecEval -> CyberMetric/SecEval and CyberMetric -> SecQA, with frozen generative models."""
import difflib,hashlib,json,random,re,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from experiments.localizer import LinearStep,TransformerStep,svd_basis
from scripts.adapter_compare_server import CompareRunner,BASE_MODEL_ID,BASE_REVISION

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
def fit(x,y,spec):
 mu=x.mean((0,1),keepdim=True);sd=x.std((0,1),keepdim=True).clamp_min(1e-6);z=(x-mu)/sd;basis=svd_basis(z,8);z=z@basis
 torch.manual_seed(42);a=LinearStep(8,16,.1);b=TransformerStep(8,16,4,.1,4)
 for m in (a,b):
  m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
  for _ in range(120):opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(z,None).mean(1),y);loss.backward();opt.step()
  m.eval()
 return a,b,mu,sd,basis
def scores(head,x):
 a,b,mu,sd,basis=head
 with torch.inference_mode():
  z=((x-mu)/sd)@basis;return (torch.sigmoid(a(z,None).mean(1))+torch.sigmoid(b(z,None).mean(1)))/2
def metrics(rows):
 return {'n':len(rows),'base_correct':sum(x['base']['correct'] for x in rows),'cyber_correct':sum(x['cyber']['correct'] for x in rows),'router_correct':sum(x['router_correct'] for x in rows),'oracle_correct':sum(x['base']['correct'] or x['cyber']['correct'] for x in rows),'switches':sum(x['switched'] for x in rows),'useful_switches':sum(x['switched'] and not x['base']['correct'] and x['cyber']['correct'] for x in rows),'harmful_switches':sum(x['switched'] and x['base']['correct'] and not x['cyber']['correct'] for x in rows),'base_errors_flagged':sum(x['switched'] and not x['base']['correct'] for x in rows),'base_errors_total':sum(not x['base']['correct'] for x in rows),'base_invalid':sum(x['base']['parsed_answer'] is None for x in rows),'cyber_invalid':sum(x['cyber']['parsed_answer'] is None for x in rows)}
def main():
 src=ROOT/'data/benchmarks/seceval/questions.json';oldpath=ROOT/'results/cybermetric80_checkpoint2500.json';old=json.loads(oldpath.read_text(encoding='utf8'));sec=torch.load(ROOT/'results/cyber_router_secqa_features.pt',weights_only=False);cyb=torch.load(ROOT/'results/cyber_router_v1_features.pt',weights_only=False)
 assert old['complete'] and len(old['cases'])==80 and cyb['spec']['data_source_sha256']==sha(oldpath)
 assert sec['spec']['training_dataset_sha256']==sha(ROOT/'data/benchmarks/secqa/secqa_v2_test.csv')
 out=ROOT/'results/cyber_router_crossdataset_v1.json';side=ROOT/'results/cyber_router_crossdataset_v1_features.pt';heads=ROOT/'results/cyber_router_crossdataset_v1_heads.pt'
 if any(x.exists() for x in (out,side,heads)):raise RuntimeError('Refusing to overwrite')
 benchmark_questions=[norm(c['question']) for c in old['cases']]
 import csv
 secqa_questions=[norm(x['Question']) for x in csv.DictReader((ROOT/'data/benchmarks/secqa/secqa_v2_test.csv').open(encoding='utf-8-sig',newline=''))]
 pool=json.loads(src.read_text(encoding='utf8')); random.Random(42).shuffle(pool)
 selected=[];discard=[];seen=set()
 for item in pool:
  answer=str(item.get('answer','')).strip().upper();choices=item.get('choices',[]);q=item.get('question','');n=norm(q)
  if answer not in ['A','B','C','D'] or len(choices)!=4 or not q:continue
  if n in seen:continue
  if any(difflib.SequenceMatcher(None,n,t).ratio()>=.85 for t in benchmark_questions+secqa_questions):discard.append(item['id']);continue
  formatted={}
  for letter,choice in zip('ABCD',choices):
   match=re.match(r'^\s*'+letter+r'\s*[:.)]\s*(.+)$',choice,flags=re.S|re.I)
   if not match:break
   formatted[letter]=match.group(1).strip()
  if len(formatted)!=4:continue
  seen.add(n);selected.append({'id':item['id'],'question':q,'answers':formatted,'expected':answer})
  if len(selected)==240:break
 assert len(selected)==240
 train=selected[:200];test=selected[200:]
 spec={'source':'XuanwuAI/SecEval questions.json','source_revision':'205dab7b0888a06f4b53ca7d9c7093e1326683e1','source_sha256':sha(src),'selection':'seed 42 shuffle, first 240 valid unique single-answer items after overlap filtering; first 200 train, final 40 test','excluded_high_string_overlap_ids':discard,'train_ids':[r['id'] for r in train],'test_ids':[r['id'] for r in test],'settings':old['settings'],'prompt_template':old['prompt_template'],'probe':'linear plus Transformer mean sigmoid; 4 final answer-token states; train-only standardization and 8d SVD','epochs':120,'seed':42,'threshold':0.5,'limitations':'Exploratory research; public datasets and earlier evaluated CyberMetric and SecQA, unknown contamination from model pretraining and adapter data. Similarity filtering does not prove semantic disjointness. SecEval and SecQA are public benchmark test items repurposed for training or evaluation. No multi-step CGR bridge test.'}
 runner=CompareRunner(ROOT/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu');runner._load();model=runner._model;tok=runner._tokenizer
 folder=ROOT/'notebooks/cyber 25000';assert sha(folder/'adapter_model.safetensors')==old['adapter_sha256'];model.load_adapter(str(folder),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
 layers=[m for m in model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')];assert len(layers)==112
 def inputs(row):return runner._inputs(old['prompt_template'].format(question=row['question'],**row['answers']))
 def generate(row,arm):
  inp=inputs(row);r=runner._generate(inp,old['settings'],arm=='cyber',adapter_name='cyber')
  match=re.match(r'^\s*ANSWER\s*:\s*([ABCD])\b',r['text'],re.I);r['parsed_answer']=match.group(1).upper() if match else None;r['correct']=r['parsed_answer']==row['expected'];return r
 def features_for(row):
  inp=inputs(row)
  with model.disable_adapter(),torch.inference_mode():
   assert all(m.disable_adapters for m in layers);torch.manual_seed(42)
   s=model.generate(**inp,max_new_tokens=12,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
   gen=s[0,inp['input_ids'].shape[-1]:];response=tok.decode(gen,skip_special_tokens=True).strip();mt=re.match(r'^\s*ANSWER\s*:\s*([ABCD])\b',response,re.I);ans=mt.group(1).upper() if mt else None
   ids=gen.tolist()
   while ids and ids[-1]==tok.eos_token_id:ids.pop()
   if not ids:raise ValueError('no non-EOS answer token')
   full=torch.cat([inp['input_ids'],torch.tensor([ids])],dim=1);h=model(input_ids=full,attention_mask=torch.ones_like(full),output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[-1][0,-min(4,len(ids)):,:].detach().float().cpu()
   if h.shape[0]<4:h=torch.cat([h[:1].repeat(4-h.shape[0],1),h],0)
  return h,{'text':response,'parsed_answer':ans,'correct':ans==row['expected']}
 results={'spec':spec,'complete':False,'created_at':datetime.now(timezone.utc).isoformat()}
 def save():out.write_text(json.dumps(results,indent=2),encoding='utf8')
 feature_list=[];base_train=[]
 for i,row in enumerate(train,1):
  h,b=features_for(row);feature_list.append(h);base_train.append({'id':row['id'],'base':b})
  if i%20==0:print(f'SecEval training {i}/200',flush=True)
 X=torch.stack(feature_list);Y=torch.tensor([float(v['base']['correct']) for v in base_train]);assert 0<Y.sum()<200
 head=fit(X,Y,spec);torch.save({'features':X,'training_base':base_train,'spec':spec},side)
 torch.save({'linear':head[0].state_dict(),'transformer':head[1].state_dict(),'mean':head[2],'std':head[3],'basis':head[4],'spec':spec},heads)
 results['seceval_training']={'n':len(Y),'base_correct':int(Y.sum()),'base_incorrect':int(len(Y)-Y.sum())};save()
 # Holdout SecEval: generate both arms, then apply the already-frozen head.
 secrows=[];xs=[]
 for i,row in enumerate(test,1):
  h,base=features_for(row);cyber=generate(row,'cyber');xs.append(h);secrows.append({'id':row['id'],'question':row['question'],'answers':row['answers'],'expected':row['expected'],'base':base,'cyber':cyber})
  if i%10==0:print(f'SecEval held-out paired generation {i}/40',flush=True)
 for row,p in zip(secrows,scores(head,torch.stack(xs))):
  row['router_score']=float(p);row['switched']=float(p)<.5;row['router_correct']=row['cyber']['correct'] if row['switched'] else row['base']['correct']
 results['seceval_heldout']={'metrics':metrics(secrows),'cases':secrows};save()
 # The frozen CyberMetric features and independently generated base/cyber responses are from the earlier recorded run.
 cybrows=[]
 for c,p in zip(old['cases'],scores(head,cyb['features'])):
  r={'index':c['index'],'base':c['base'],'cyber':c['cyber'],'router_score':float(p),'switched':float(p)<.5};r['router_correct']=r['cyber']['correct'] if r['switched'] else r['base']['correct'];cybrows.append(r)
 results['seceval_to_cybermetric']={'metrics':metrics(cybrows),'cases':cybrows};save();print('SecEval -> CyberMetric',json.dumps(results['seceval_to_cybermetric']['metrics']),flush=True)
 # Inverse experiment: fit using all CyberMetric features and labels; SecQA examples have never trained these heads.
 inverse=fit(cyb['features'],cyb['labels'],spec)
 inverse_path=ROOT/'results/cyber_router_cybermetric_to_secqa_heads.pt';torch.save({'linear':inverse[0].state_dict(),'transformer':inverse[1].state_dict(),'mean':inverse[2],'std':inverse[3],'basis':inverse[4],'spec':spec},inverse_path)
 secqa_rows=[];scr=scores(inverse,sec['features'])
 for i,(ex,p) in enumerate(zip(sec['examples'],scr),1):
  # Decode question and options directly from the original recorded prompt; exact input used for SecQA feature collection.
  inp=runner._inputs(ex['prompt']);result=runner._generate(inp,old['settings'],True,adapter_name='cyber')
  mt=re.match(r'^\s*ANSWER\s*:\s*([ABCD])\b',result['text'],re.I);ans=mt.group(1).upper() if mt else None;result['parsed_answer']=ans;result['correct']=ans==ex['expected']
  base={'text':ex['response'],'parsed_answer':ex['parsed_answer'],'correct':ex['correct']}
  row={'source_row':ex['source_row'],'expected':ex['expected'],'base':base,'cyber':result,'router_score':float(p),'switched':float(p)<.5};row['router_correct']=result['correct'] if row['switched'] else base['correct'];secqa_rows.append(row)
  if i%10==0:print(f'CyberMetric -> SecQA paired generation {i}/100',flush=True)
 results['cybermetric_to_secqa']={'training_questions':80,'metrics':metrics(secqa_rows),'cases':secqa_rows};results['complete']=True;results['finished_at']=datetime.now(timezone.utc).isoformat();save();print(json.dumps({k:results[k]['metrics'] for k in ('seceval_heldout','seceval_to_cybermetric','cybermetric_to_secqa')},indent=2),flush=True)
if __name__=='__main__':main()