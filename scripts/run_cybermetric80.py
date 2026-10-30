import hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.adapter_compare_server import CompareRunner, BASE_MODEL_ID, BASE_REVISION
from transformers import AutoTokenizer
root=Path(__file__).resolve().parents[1]
source=root/'data/benchmarks/cybermetric80/CyberMetric-80-v1.json'
questions=json.loads(source.read_text(encoding='utf-8'))['questions']
assert len(questions)==80 and all(set(q['answers'])==set('ABCD') and q['solution'] in 'ABCD' for q in questions)
folder=root/'notebooks/cyber 25000'
manifest=json.loads((folder/'run_manifest.json').read_text())
config=json.loads((folder/'adapter_config.json').read_text())
state=json.loads((folder/'trainer_state.json').read_text())
assert manifest['base_model_id']==BASE_MODEL_ID==config['base_model_name_or_path'] and manifest['base_revision']==BASE_REVISION and state['global_step']==2500
out=root/'results/cybermetric80_checkpoint2500.json'
if out.exists(): raise RuntimeError('Results already exist; preserve this run')
runner=CompareRunner(root/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu')
runner._load()
runner._model.load_adapter(str(folder),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
runner._model.set_adapter('cyber',inference_mode=True)
runner._model.base_model.enable_adapter_layers()
layers=[m for m in runner._model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')]
assert len(layers)==112 and all(not m.disable_adapters and list(m.active_adapters)==['cyber'] and 'cyber' in m.lora_A for m in layers)
params=[p for n,p in runner._model.named_parameters() if '.cyber.' in n]
assert len(params)==224 and all(runner._torch.isfinite(p).all().item() for p in params)
tok=AutoTokenizer.from_pretrained(str(folder),local_files_only=True)
settings={'seed':42,'do_sample':False,'max_new_tokens':12}
rec={'benchmark':'CyberMetric-80-v1','source':'https://github.com/cybermetric/CyberMetric','dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'base_model':BASE_MODEL_ID,'base_revision':BASE_REVISION,'adapter_step':2500,'adapter_sha256':hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest(),'settings':settings,'device':runner._device,'dtype':runner._dtype_name,'cpu_threads':runner._torch.get_num_threads(),'runtime':{p:version(p) for p in ('torch','transformers','peft','accelerate')},'prompt_template':'Answer the following multiple choice question. The entire content of your response should be of the following format: ANSWER: LETTER, where LETTER is one of A, B, C, D.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}','parsing':'First line must start with ANSWER: followed by a single A-D; otherwise invalid and counted incorrect.','limitations':'Zero-shot prompt adapted from Inspect CyberMetric. Public benchmark; overlap with pretraining or adapter training has not been checked. Multiple choice knowledge, not an operational security or CGR benchmark.','started_at':datetime.now(timezone.utc).isoformat(),'cases':[],'complete':False}
for i,q in enumerate(questions):
 prompt=rec['prompt_template'].format(question=q['question'],**q['answers'])
 inputs=runner._inputs(prompt)
 ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 assert ids==inputs['input_ids'][0].tolist()
 case={'index':i+1,'question':q['question'],'answers':q['answers'],'expected':q['solution'],'prompt_token_ids':ids}
 for arm in ('base','cyber'):
  if arm=='base':
   runner._model.base_model.disable_adapter_layers()
   assert all(m.disable_adapters for m in layers)
  result=runner._generate(inputs,settings,arm=='cyber',adapter_name='cyber')
  match=re.match(r'^\s*ANSWER\s*:\s*([ABCD])\b',result['text'],re.I)
  result['parsed_answer']=match.group(1).upper() if match else None
  result['correct']=result['parsed_answer']==q['solution']
  case[arm]=result
 rec['cases'].append(case)
 rec['counts']={arm:{'correct':sum(c[arm]['correct'] for c in rec['cases']),'invalid':sum(c[arm]['parsed_answer'] is None for c in rec['cases'])} for arm in ('base','cyber')}
 out.write_text(json.dumps(rec,indent=2),encoding='utf-8')
 print(f"{i+1}/80 base {rec['counts']['base']} cyber {rec['counts']['cyber']}",flush=True)
rec['complete']=True
rec['finished_at']=datetime.now(timezone.utc).isoformat()
out.write_text(json.dumps(rec,indent=2),encoding='utf-8')