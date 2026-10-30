"""Apply the previously fitted cyber router to all 80; split results remain separate."""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.localizer import LinearStep,TransformerStep
from scripts.adapter_compare_server import CompareRunner,BASE_MODEL_ID,BASE_REVISION

def main():
  old=json.loads((ROOT/'results/cybermetric80_checkpoint2500.json').read_text(encoding='utf8'))
  previous=json.loads((ROOT/'results/cyber_router_v1.json').read_text(encoding='utf8'))
  features=torch.load(ROOT/'results/cyber_router_v1_features.pt',weights_only=False)
  heads=torch.load(ROOT/'results/cyber_router_v1_heads.pt',weights_only=False)
  assert previous['complete'] and previous['spec']==features['spec']==heads['spec']
  assert hashlib.sha256((ROOT/'results/cybermetric80_checkpoint2500.json').read_bytes()).hexdigest()==heads['spec']['data_source_sha256']
  out=ROOT/'results/cyber_router_v1_full80.json'
  if out.exists(): raise RuntimeError('Refusing to overwrite existing result')
  train=set(heads['spec']['train_indices']);test=set(heads['spec']['test_indices']);assert len(train)==len(test)==40 and not train&test and train|test==set(range(1,81))
  a=LinearStep(8,16,.1);b=TransformerStep(8,16,4,.1,4);a.load_state_dict(heads['linear_state']);b.load_state_dict(heads['transformer_state']);a.eval();b.eval()
  x=(features['features']-heads['mean'])/heads['std'];x=x@heads['svd_basis']
  with torch.inference_mode():
   p=(torch.sigmoid(a(x,None).mean(1))+torch.sigmoid(b(x,None).mean(1)))/2
  previous_rows={r['index']:r for r in previous['test_cases']}
  for i in test: assert abs(float(p[i-1])-previous_rows[i]['p_correct_score'])<1e-6
  assert old['base_model']==BASE_MODEL_ID and old['base_revision']==BASE_REVISION
  folder=ROOT/'notebooks/cyber 25000';assert hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest()==previous['cyber_adapter_sha256']
  runner=CompareRunner(ROOT/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu');runner._load()
  runner._model.load_adapter(str(folder),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
  layers=[m for m in runner._model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')];assert len(layers)==112
  data={'complete':False,'started_at':datetime.now(timezone.utc).isoformat(),'source_router_sha256':hashlib.sha256((ROOT/'results/cyber_router_v1.json').read_bytes()).hexdigest(),'source_heads_sha256':hashlib.sha256((ROOT/'results/cyber_router_v1_heads.pt').read_bytes()).hexdigest(),'split':'original 40 train / 40 test, unchanged; full-80 total includes in-sample training rows','threshold':0.5,'limitations':'Training answers are in-sample; full total is not held-out accuracy. Original benchmark was previously examined. No DAG or bridge timing measured.','cases':[]}
  for i,c in enumerate(old['cases'],1):
   score=float(p[i-1]);switch=score<.5
   row={'index':i,'split':'train' if i in train else 'test','router_score':score,'switched':switch,'base_correct':c['base']['correct'],'cyber_correct':c['cyber']['correct'],'router_correct':c['cyber']['correct'] if switch else c['base']['correct']}
   if switch:
    prompt=old['prompt_template'].format(question=c['question'],**c['answers']); inputs=runner._inputs(prompt);assert inputs['input_ids'][0].tolist()==c['prompt_token_ids']
    fresh=runner._generate(inputs,old['settings'],True,adapter_name='cyber')
    assert all(not m.disable_adapters and list(m.active_adapters)==['cyber'] for m in layers)
    if fresh['text']!=c['cyber']['text']: raise RuntimeError(f'Fresh adapter output mismatch on {i}')
    row['fresh_cyber_output']=fresh
   data['cases'].append(row)
   out.write_text(json.dumps(data,indent=2),encoding='utf8')
   if i%10==0: print(f'{i}/80 complete',flush=True)
  def metrics(rows):
   return {'n':len(rows),'base_correct':sum(r['base_correct'] for r in rows),'cyber_correct':sum(r['cyber_correct'] for r in rows),'router_correct':sum(r['router_correct'] for r in rows),'oracle_correct':sum(r['base_correct'] or r['cyber_correct'] for r in rows),'switches':sum(r['switched'] for r in rows),'useful_switches':sum(r['switched'] and not r['base_correct'] and r['cyber_correct'] for r in rows),'harmful_switches':sum(r['switched'] and r['base_correct'] and not r['cyber_correct'] for r in rows),'probe_correct':sum((r['router_score']>=.5)==r['base_correct'] for r in rows)}
  data['metrics']={s:metrics([r for r in data['cases'] if s=='all' or r['split']==s]) for s in ('train','test','all')};data['complete']=True;data['finished_at']=datetime.now(timezone.utc).isoformat();out.write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps(data['metrics'],indent=2),flush=True)
if __name__=='__main__':main()