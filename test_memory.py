"""Exploratory paired CyberMetric experience-memory test; no training or DAGs."""
import hashlib, json, random, re
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
from experience_memory import ExperienceMemory
from scripts.adapter_compare_server import CompareRunner, BASE_MODEL_ID, BASE_REVISION

def main():
    ROOT=Path(__file__).resolve().parent
    old=json.loads((ROOT/'results/cybermetric80_checkpoint2500.json').read_text(encoding='utf-8'))
    assert old['complete'] and len(old['cases'])==80 and old['base_revision']==BASE_REVISION
    folder=ROOT/'notebooks/cyber 25000'
    weight_hash=hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert weight_hash==old['adapter_sha256'] and old['base_model']==BASE_MODEL_ID
    result_path=ROOT/'results/cybermetric80_experience_memory_exploratory.json'
    if result_path.exists(): raise RuntimeError('Refusing to overwrite existing results')
    print('Loading embedding model and in-memory Chroma',flush=True)
    memory=ExperienceMemory(model_name=str(ROOT/'.experience_memory_models/all-MiniLM-L6-v2'))
    assert memory.get_rules('test') == ''
    memory.add_rule('Cybersecurity multiple choice exam questions require a letter answer in a prescribed format', 'The model ignored the ANSWER: LETTER format and copied an option instead.', 'Output only ANSWER: followed by the letter A, B, C, or D.', scope='cyber_test', tier=2)
    memory.add_rule('Python programming database queries retrieve a record by ID', 'SQL injection from string concatenation', 'Use parameterized SQL queries.', scope='database', tier=2)
    memory.add_rule('Write a responsive HTML webpage with an email signup form', 'Missing accessible form label', 'Give the email input an associated label.', scope='webpage', tier=2)
    invalid=[c['index'] for c in old['cases'] if c['cyber']['parsed_answer'] is None]
    assert invalid==[2,22]
    heldout=sorted(random.Random(42).sample([i for i in range(1,81) if i not in invalid],18))
    selected=sorted(invalid+heldout)
    print('Loading pinned base and cyber checkpoint',flush=True)
    runner=CompareRunner(ROOT/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu')
    runner._load()
    runner._model.load_adapter(str(folder),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
    runner._model.set_adapter('cyber',inference_mode=True);runner._model.base_model.enable_adapter_layers()
    layers=[m for m in runner._model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')]
    assert len(layers)==112 and all(not m.disable_adapters and list(m.active_adapters)==['cyber'] and 'cyber' in m.lora_A for m in layers)
    record={'experiment':'cybermetric80_experience_memory_exploratory_v1','created_at':datetime.now(timezone.utc).isoformat(),'complete':False,'base_model':BASE_MODEL_ID,'base_revision':BASE_REVISION,'adapter_step':2500,'adapter_sha256':weight_hash,'baseline_file':str(ROOT/'results/cybermetric80_checkpoint2500.json'),'baseline_sha256':hashlib.sha256((ROOT/'results/cybermetric80_checkpoint2500.json').read_bytes()).hexdigest(),'dataset_sha256':old['dataset_sha256'],'settings':old['settings'],'device':runner._device,'dtype':runner._dtype_name,'versions':{k:version(k) for k in ['chromadb','sentence-transformers','torch','transformers','peft']},'embedding_model':'sentence-transformers/all-MiniLM-L6-v2','selection':{'targeted_failure_indices':invalid,'random_seed':42,'random_other_indices':heldout},'retrieval':'cosine similarity, top 1 of eligible global and cyber_test rules; retrieval query is the original benchmark prompt','parsing':old['parsing'],'limitations':'Exploratory: rule was authored after observing benchmark failures; other cases were previously evaluated. Public benchmark overlap with training unknown. No independent held-out result or causal proof retrieval outperforms direct instruction. Baseline taken from earlier run.','cases':[]}
    for j,index in enumerate(selected,1):
     case=old['cases'][index-1]; assert case['index']==index
     prompt=old['prompt_template'].format(question=case['question'],**case['answers'])
     retrieved=memory.retrieve(prompt,top_k=1,scope='cyber_test')
     tag=memory.get_rules(prompt,top_k=1,scope='cyber_test')
     assert tag=='[SYSTEM MEMORY: '+retrieved[0]['rule']+']'
     sent=tag+'\n'+prompt
     inputs=runner._inputs(sent)
     item={'index':index,'group':'targeted' if index in invalid else 'other_sample','expected':case['expected'],'retrieved':retrieved,'memory_tag':tag,'prompt_sent':sent,'prompt_token_ids':inputs['input_ids'][0].tolist(),'baseline':{'base':case['base'],'cyber':case['cyber']},'memory':{}}
     for arm in ('base','cyber'):
      if arm=='base':
       runner._model.base_model.disable_adapter_layers(); assert all(m.disable_adapters for m in layers)
      res=runner._generate(inputs,old['settings'],arm=='cyber',adapter_name='cyber')
      match=re.match(r'^\s*ANSWER\s*:\s*([ABCD])\b',res['text'],re.I)
      res['parsed_answer']=match.group(1).upper() if match else None
      res['correct']=res['parsed_answer']==case['expected']
      item['memory'][arm]=res
     record['cases'].append(item)
     result_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
     print(f'{j}/{len(selected)} question {index}; retrieved {retrieved[0]["rule"]}',flush=True)
    record['complete']=True; record['finished_at']=datetime.now(timezone.utc).isoformat()
    result_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Saved',result_path,flush=True)

if __name__ == "__main__":
    main()
