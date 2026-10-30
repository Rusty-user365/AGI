"""Live base -> hidden-state router -> fresh cyber prefill on CyberMetric-80."""
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.localizer import LinearStep,TransformerStep
from scripts.adapter_compare_server import CompareRunner,BASE_MODEL_ID,BASE_REVISION

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 oldpath=ROOT/'results/cybermetric80_checkpoint2500.json'; refpath=ROOT/'results/cyber_router_crossdataset_v1.json';hpath=ROOT/'results/cyber_router_crossdataset_v1_heads.pt'
 old=json.loads(oldpath.read_text(encoding='utf8'));ref=json.loads(refpath.read_text(encoding='utf8'));head=torch.load(hpath,weights_only=False)
 assert old['complete'] and ref['complete'] and old['base_model']==BASE_MODEL_ID and old['base_revision']==BASE_REVISION
 assert sha(ROOT/'notebooks/cyber 25000/adapter_model.safetensors')==old['adapter_sha256']
 assert head['spec']['source_sha256']==ref['spec']['source_sha256'] and len(old['cases'])==80
 out=ROOT/'results/cyber_router_live_cybermetric80_v3.json'
 if out.exists():raise RuntimeError('Refusing to overwrite an existing result')
 a=LinearStep(8,16,.1);b=TransformerStep(8,16,4,.1,4);a.load_state_dict(head['linear']);b.load_state_dict(head['transformer']);a.eval();b.eval()
 runner=CompareRunner(ROOT/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu');runner._load();model=runner._model;tok=runner._tokenizer
 model.load_adapter(str(ROOT/'notebooks/cyber 25000'),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
 layers=[m for m in model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')];assert len(layers)==112
 saved={x['index']:x for x in ref['seceval_to_cybermetric']['cases']}
 record={'started_at':datetime.now(timezone.utc).isoformat(),'complete':False,'source_benchmark_sha256':sha(oldpath),'source_offline_router_sha256':sha(refpath),'heads_sha256':sha(hpath),'base_model':BASE_MODEL_ID,'base_revision':BASE_REVISION,'adapter_sha256':old['adapter_sha256'],'settings':old['settings'],'threshold':.5,'method':'Regenerate base with no adapter, recompute answer-token features; choose using frozen heads. On switch generate under cyber from entire prompt with a fresh empty cache; compare with saved standalone. No answer key is used in routing.','limitations':'Independent single-question routing, not a DAG, and does not test commit verification or base bridge replay. Public benchmark previously inspected and potential training contamination unknown.','cases':[]}
 for i,c in enumerate(old['cases'],1):
  start=time.perf_counter();prompt=old['prompt_template'].format(question=c['question'],**c['answers']);inp=runner._inputs(prompt);assert inp['input_ids'][0].tolist()==c['prompt_token_ids']
  with model.disable_adapter(),torch.inference_mode():
   assert all(m.disable_adapters for m in layers);torch.manual_seed(old['settings']['seed'])
   seq=model.generate(**inp,max_new_tokens=old['settings']['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
   gen=seq[0,inp['input_ids'].shape[-1]:];base=tok.decode(gen,skip_special_tokens=True).strip()
   if base!=c['base']['text']:raise RuntimeError(f'Base output differs on {i}')
   generated=gen.tolist()
   while generated and generated[-1]==tok.eos_token_id:generated.pop()
   if not generated:raise RuntimeError('Missing generated answer tokens')
   full=torch.cat([inp['input_ids'],torch.tensor([generated])],dim=1)
   h=model(input_ids=full,attention_mask=torch.ones_like(full),output_hidden_states=True,use_cache=False,return_dict=True).hidden_states[-1][0,-min(4,len(generated)):,:].detach().float().cpu()
   if len(h)<4:h=torch.cat([h[:1].repeat(4-len(h),1),h],0)
   z=((h.unsqueeze(0)-head['mean'])/head['std'])@head['basis']
   prob=float(((torch.sigmoid(a(z,None).mean(1))+torch.sigmoid(b(z,None).mean(1)))/2).item())
  switched=prob<.5
  prior=saved[i]
  if abs(prob-prior['router_score'])>1e-5 or switched!=prior['switched']:raise RuntimeError(f'Router reproduction differs on {i}')
  row={'index':i,'base_text':base,'base_matches_saved':True,'router_score':prob,'saved_router_score':prior['router_score'],'switched':switched,'switch_matches_saved':True,'base_correct':c['base']['correct'],'cyber_reference_correct':c['cyber']['correct']}
  if switched:
   calls=[]
   def capture(module,args,kwargs):
    if calls:return
    ids=kwargs.get('input_ids'); cache=kwargs.get('past_key_values')
    past_len=0 if cache is None else cache.get_seq_length()
    calls.append({'input_ids':ids.detach().cpu().tolist()[0],'cache_length_before':past_len,'cache_type':type(cache).__name__ if cache is not None else None})
   hook=model.base_model.model.register_forward_pre_hook(capture,with_kwargs=True)
   try: cyber=runner._generate(inp,old['settings'],True,adapter_name='cyber')
   finally:hook.remove()
   if len(calls)!=1 or calls[0]['input_ids']!=c['prompt_token_ids'] or calls[0]['cache_length_before']!=0:raise RuntimeError(f'Fresh cyber prefill invariant failed on {i}: {calls!r}; expected length {len(c["prompt_token_ids"])}')
   if any(m.disable_adapters or list(m.active_adapters)!=['cyber'] for m in layers):raise RuntimeError('Wrong active adapter')
   if cyber['text']!=c['cyber']['text']:raise RuntimeError(f'Cyber output differs on {i}')
   row.update({'cyber_text':cyber['text'],'cyber_matches_saved':True,'cyber_first_forward':{'input_tokens':len(calls[0]['input_ids']),'full_prompt_matches':True,'cache_length_before':calls[0]['cache_length_before'],'cache_type':calls[0]['cache_type']},'cyber_generation_seconds':cyber['latency_seconds']})
  row['router_correct']=c['cyber']['correct'] if switched else c['base']['correct'];row['elapsed_seconds']=time.perf_counter()-start
  record['cases'].append(row);out.write_text(json.dumps(record,indent=2),encoding='utf8')
  if i%10==0:print(f'Live questions {i}/80, switches {sum(r["switched"] for r in record["cases"])}',flush=True)
 rows=record['cases'];record['metrics']={'n':len(rows),'base_correct':sum(r['base_correct'] for r in rows),'cyber_correct':sum(r['cyber_reference_correct'] for r in rows),'router_correct':sum(r['router_correct'] for r in rows),'oracle_correct':sum(r['base_correct'] or r['cyber_reference_correct'] for r in rows),'switches':sum(r['switched'] for r in rows),'useful_switches':sum(r['switched'] and not r['base_correct'] and r['cyber_reference_correct'] for r in rows),'harmful_switches':sum(r['switched'] and r['base_correct'] and not r['cyber_reference_correct'] for r in rows),'base_output_matches':sum(r['base_matches_saved'] for r in rows),'decisions_match':sum(r['switch_matches_saved'] for r in rows),'cyber_output_matches':sum(r.get('cyber_matches_saved',False) for r in rows),'empty_cache_prefills':sum(r.get('cyber_first_forward',{}).get('cache_length_before')==0 for r in rows)}
 record['complete']=True;record['finished_at']=datetime.now(timezone.utc).isoformat();out.write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record['metrics'],indent=2),flush=True)
if __name__=='__main__':main()
