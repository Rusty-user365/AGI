import hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.adapter_compare_server import CompareRunner, BASE_MODEL_ID, BASE_REVISION
from transformers import AutoTokenizer
root = Path(__file__).resolve().parents[1]
folder = root / 'notebooks' / 'cyber 25000'
manifest = json.loads((folder/'run_manifest.json').read_text())
config = json.loads((folder/'adapter_config.json').read_text())
state = json.loads((folder/'trainer_state.json').read_text())
assert manifest['base_model_id'] == BASE_MODEL_ID == config['base_model_name_or_path']
assert manifest['base_revision'] == BASE_REVISION and state['global_step'] == 2500
cases = [
 {'id':'phishing','prompt':'A finance employee receives an unexpected email asking them to sign in through a shortened link to avoid payroll suspension. What is the safest first action? Reply with only A, B, or C.\nA. Enter their password to verify the request.\nB. Independently contact the finance or security team through a known channel and report the message.\nC. Forward the link to coworkers to see if it works.','expected':'B'},
 {'id':'sql','prompt':'A Python web application builds a SQL query by concatenating a username into the query string. Which change best prevents SQL injection? Reply with only A, B, or C.\nA. Use parameterized queries with bound values.\nB. Hide the database error message.\nC. Base64-encode the username before concatenation.','expected':'A'},
 {'id':'credential','prompt':'A production API key was accidentally committed to a public repository. What is the best immediate response? Reply with only A, B, or C.\nA. Delete the latest line and assume the key is safe.\nB. Revoke or rotate the exposed key and investigate its use.\nC. Rename the repository.','expected':'B'},
 {'id':'mfa','prompt':'An attacker obtained a user password. Which measure most directly reduces the chance that password alone can be used to sign in? Reply with only A, B, or C.\nA. Increase the browser cache size.\nB. Disable audit logs.\nC. Require phishing-resistant multi-factor authentication.','expected':'C'},
]
output = root/'results'/'cyber_checkpoint_2500_smoke_test.json'
runner = CompareRunner(root/'notebooks'/'coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter'/'checkpoint-1000', 'cpu')
runner._load()
runner._model.load_adapter(str(folder), adapter_name='cyber', is_trainable=False, device_map={'':'cpu'})
runner._model.set_adapter('cyber',inference_mode=True)
runner._model.base_model.enable_adapter_layers()
layers=[m for m in runner._model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')]
assert len(layers)==112 and all(not m.disable_adapters and list(m.active_adapters)==['cyber'] and 'cyber' in m.lora_A for m in layers)
params=[p for n,p in runner._model.named_parameters() if '.cyber.' in n]
assert len(params)==224 and all(runner._torch.isfinite(p).all().item() for p in params)
trained = AutoTokenizer.from_pretrained(str(folder),local_files_only=True)
record={'created_at':datetime.now(timezone.utc).isoformat(),'base_model':BASE_MODEL_ID,'base_revision':BASE_REVISION,'adapter_step':2500,'adapter_path':str(folder),'weights_sha256':hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest(),'settings':{'seed':42,'do_sample':False,'max_new_tokens':32},'device':runner._device,'dtype':runner._dtype_name,'active_layers':len(layers),'finite_tensors':len(params),'cases':[],'limitations':'Small newly authored smoke test; not proven disjoint from training data, not a safety benchmark, and not a CGR switching test.'}
import re
for case in cases:
 inputs=runner._inputs(case['prompt'])
 ids=trained.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 assert ids==inputs['input_ids'][0].tolist()
 item={**case,'prompt_token_ids':ids}
 for arm in ('base','cyber'):
  if arm=='base':
   runner._model.base_model.disable_adapter_layers()
   assert all(m.disable_adapters for m in layers)
  result=runner._generate(inputs,record['settings'],arm=='cyber',adapter_name='cyber')
  match=re.match(r'^\s*([ABC])(?:\b|[.)])',result['text'],re.I)
  result['parsed_answer']=match.group(1).upper() if match else None
  result['correct']=result['parsed_answer']==case['expected']
  item[arm]=result
  print(case['id'],arm,json.dumps(result),flush=True)
 record['cases'].append(item)
 output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Saved',output,flush=True)