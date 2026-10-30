"""Train cyber error detectors on SecQA v2 and evaluate on CyberMetric-80."""
import csv,difflib,hashlib,json,random,re,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from experiments.localizer import LinearStep,TransformerStep,svd_basis
from scripts.adapter_compare_server import CompareRunner,BASE_MODEL_ID,BASE_REVISION

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(s): return re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
def main():
 src=ROOT/'data/benchmarks/secqa/secqa_v2_test.csv';benchmark=ROOT/'results/cybermetric80_checkpoint2500.json'
 rows=list(csv.DictReader(src.open(encoding='utf-8-sig',newline='')));old=json.loads(benchmark.read_text(encoding='utf-8'))
 assert len(rows)==100 and len(old['cases'])==80 and old['complete'] and old['base_model']==BASE_MODEL_ID and old['base_revision']==BASE_REVISION
 assert all(r['Answer'] in 'ABCD' and all(r.get(k) for k in ('Question','A','B','C','D')) for r in rows)
 # Normalize and filter high string-similarity, without using CyberMetric answers.
 excluded=[];train=[]
 for i,row in enumerate(rows):
  scores=[difflib.SequenceMatcher(None,norm(row['Question']),norm(c['question'])).ratio() for c in old['cases']]
  if max(scores)>=.85:excluded.append({'source_row':i+1,'matched_question':scores.index(max(scores))+1,'similarity':max(scores)})
  else:train.append((i+1,row))
 out=ROOT/'results/cyber_router_secqa_to_cybermetric80.json'; featfile=ROOT/'results/cyber_router_secqa_features.pt';headfile=ROOT/'results/cyber_router_secqa_heads.pt'
 if any(p.exists() for p in (out,featfile,headfile)):raise RuntimeError('Refusing to overwrite prior experiment')
 eval_tensor=torch.load(ROOT/'results/cyber_router_v1_features.pt',weights_only=False)
 assert eval_tensor['spec']['data_source_sha256']==digest(benchmark)
 X_eval=eval_tensor['features'] # Do not train or normalize using test features or labels.
 folder=ROOT/'notebooks/cyber 25000';assert digest(folder/'adapter_model.safetensors')==old['adapter_sha256']
 spec={'training_dataset':'zefang-liu/secqa secqa_v2_test.csv (originally SecQA test split, repurposed here for probe training)','training_dataset_revision':'d00a07484283be5602e2bae36dbefdaaf555a9fb','training_dataset_sha256':digest(src),'benchmark_sha256':old['dataset_sha256'],'baseline_record_sha256':digest(benchmark),'training_rows':len(train),'excluded_near_duplicates':excluded,'feature':'last four non-EOS answer tokens, final hidden layer; left repeat if fewer than four','preprocessing':'mean, std and 8-dimensional SVD fitted only on SecQA training features','architecture':'LinearStep and TransformerStep, mean over token logits then mean of sigmoid scores','width':16,'heads':4,'epochs':120,'seed':42,'threshold':0.5,'optimizer':'AdamW lr=0.001 weight_decay=0.0001','settings':old['settings'],'limitations':'Exploratory, CyberMetric-80 has been inspected previously and is not a pristine blinded holdout. Source dataset overlap checks exclude exact and >=0.85 character-similarity question matches, not semantic paraphrases. SecQA is itself a benchmark used for head training. Pretraining and adapter data contamination unknown. No multistep DAG or bridge verification.'}
 runner=CompareRunner(ROOT/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu');runner._load();model=runner._model;tok=runner._tokenizer
 model.load_adapter(str(folder),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
 layers=[m for m in model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')];assert len(layers)==112
 features=[];labels=[];examples=[]
 for j,(index,r) in enumerate(train,1):
  prompt=old['prompt_template'].format(question=r['Question'],**{x:r[x] for x in 'ABCD'})
  inputs=runner._inputs(prompt)
  with model.disable_adapter(),torch.inference_mode():
   assert all(m.disable_adapters for m in layers)
   torch.manual_seed(42)
   seq=model.generate(**inputs,max_new_tokens=old['settings']['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
   gen=seq[0,inputs['input_ids'].shape[-1]:];text=tok.decode(gen,skip_special_tokens=True).strip()
   match=re.match(r'^\s*ANSWER\s*:\s*([ABCD])\b',text,re.I);answer=match.group(1).upper() if match else None
   ids=gen.tolist()
   while ids and ids[-1]==tok.eos_token_id:ids.pop()
   if not ids:raise RuntimeError('Empty response')
   full=torch.cat([inputs['input_ids'],torch.tensor([ids],device=inputs['input_ids'].device)],dim=1)
   h=model(input_ids=full,attention_mask=torch.ones_like(full),output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[-1][0,-min(4,len(ids)):,:].detach().float().cpu()
   if h.shape[0]<4:h=torch.cat([h[:1].repeat(4-h.shape[0],1),h],0)
   features.append(h);labels.append(float(answer==r['Answer']));examples.append({'source_row':index,'prompt':prompt,'expected':r['Answer'],'response':text,'parsed_answer':answer,'correct':answer==r['Answer']})
  if j%10==0:print(f'SecQA training feature collection {j}/{len(train)}',flush=True)
 X=torch.stack(features);Y=torch.tensor(labels);assert len(Y)==len(train) and 0<Y.sum()<len(Y)
 torch.save({'features':X,'labels':Y,'examples':examples,'spec':spec},featfile)
 mu=X.mean((0,1),keepdim=True);sd=X.std((0,1),keepdim=True).clamp_min(1e-6);s=(X-mu)/sd;basis=svd_basis(s,8);z=s@basis
 torch.manual_seed(42);linear=LinearStep(8,16,.1);transformer=TransformerStep(8,16,4,.1,4)
 for m in (linear,transformer):
  opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);m.train()
  for _ in range(120):
   opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(z,None).mean(1),Y);loss.backward();opt.step()
  m.eval()
 torch.save({'linear_state':linear.state_dict(),'transformer_state':transformer.state_dict(),'mean':mu,'std':sd,'svd_basis':basis,'spec':spec},headfile)
 with torch.inference_mode():
  z_eval=((X_eval-mu)/sd)@basis; pl=torch.sigmoid(linear(z_eval,None).mean(1));pt=torch.sigmoid(transformer(z_eval,None).mean(1));p=(pl+pt)/2
 record={'spec':spec,'training_correct':int(Y.sum()),'training_incorrect':int(len(Y)-Y.sum()),'feature_sha256':digest(featfile),'heads_sha256':digest(headfile),'created_at':datetime.now(timezone.utc).isoformat(),'complete':False,'cases':[]}
 for i,c in enumerate(old['cases']):
  score=float(p[i]);switch=score<.5
  row={'index':i+1,'expected':c['expected'],'router_score':score,'linear_score':float(pl[i]),'transformer_score':float(pt[i]),'switched':switch,'base_correct':c['base']['correct'],'cyber_correct':c['cyber']['correct'],'router_correct':c['cyber']['correct'] if switch else c['base']['correct']}
  if switch:
   prompt=old['prompt_template'].format(question=c['question'],**c['answers']);inputs=runner._inputs(prompt);assert inputs['input_ids'][0].tolist()==c['prompt_token_ids']
   fresh=runner._generate(inputs,old['settings'],True,adapter_name='cyber')
   if fresh['text']!=c['cyber']['text']:raise RuntimeError(f'Cyber generation differs from reference case {i+1}')
   assert all(not m.disable_adapters and list(m.active_adapters)==['cyber'] for m in layers)
   row['fresh_cyber_output']=fresh
  record['cases'].append(row);out.write_text(json.dumps(record,indent=2),encoding='utf-8')
  if (i+1)%10==0:print(f'CyberMetric evaluation {i+1}/80',flush=True)
 t=record['cases'];record['metrics']={'n':len(t),'base_correct':sum(x['base_correct'] for x in t),'cyber_correct':sum(x['cyber_correct'] for x in t),'router_correct':sum(x['router_correct'] for x in t),'oracle_correct':sum(x['base_correct'] or x['cyber_correct'] for x in t),'switches':sum(x['switched'] for x in t),'useful_switches':sum(x['switched'] and not x['base_correct'] and x['cyber_correct'] for x in t),'harmful_switches':sum(x['switched'] and x['base_correct'] and not x['cyber_correct'] for x in t),'probe_correct':sum((x['router_score']>=.5)==x['base_correct'] for x in t),'base_errors_flagged':sum(x['switched'] and not x['base_correct'] for x in t),'base_errors_total':sum(not x['base_correct'] for x in t)}
 record['complete']=True;record['finished_at']=datetime.now(timezone.utc).isoformat();out.write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record['metrics'],indent=2),flush=True)
if __name__=='__main__':main()