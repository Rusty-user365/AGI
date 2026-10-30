"""Exploratory CyberMetric router: 40/40, fixed split and threshold, fresh cyber prefill."""
import hashlib, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from experiments.localizer import LinearStep, TransformerStep, svd_basis
from scripts.adapter_compare_server import CompareRunner, BASE_MODEL_ID, BASE_REVISION

def main():
    oldpath=ROOT/'results/cybermetric80_checkpoint2500.json'
    old=json.loads(oldpath.read_text(encoding='utf-8'))
    assert old['complete'] and len(old['cases'])==80 and old['base_model']==BASE_MODEL_ID and old['base_revision']==BASE_REVISION
    folder=ROOT/'notebooks/cyber 25000'
    whash=hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert whash==old['adapter_sha256']
    out=ROOT/'results/cyber_router_v1.json'; features_path=ROOT/'results/cyber_router_v1_features.pt'
    if out.exists() or features_path.exists(): raise RuntimeError('Refusing to overwrite previous router result')
    # Freeze all experimental choices before collecting features.
    indices=list(range(80));random.Random(42).shuffle(indices); train=indices[:40];test=indices[40:]
    spec={'split_seed':42,'train_indices':[i+1 for i in train],'test_indices':[i+1 for i in test], 'epochs':120,'seed':42,'p_correct_threshold':0.5,'feature':'last four generated non-EOS token hidden states from final layer, repeat first if fewer than four','standardization':'train tokens only','projection':'train-only SVD, 8 dimensions','ensemble':'mean of sigmoid probabilities of LinearStep and TransformerStep; token logits averaged within each binary classifier', 'width':16,'heads':4,'optimizer':'AdamW lr=0.001 weight_decay=0.0001','dropout':0.1,'data_source_sha256':hashlib.sha256(oldpath.read_bytes()).hexdigest(),'baseline_generation_settings':old['settings'],'limitations':'Exploratory public 80-question benchmark, previously inspected; 40 test items not used to fit probe but not a pristine blind holdout. Router scores are uncalibrated; labels are exact answer correctness. No multi-step DAG or rollback speedup is tested.'}
    runner=CompareRunner(ROOT/'notebooks/coder_qwen3_0.6b_q_mlp_r16_v1___Coder Adapter/checkpoint-1000','cpu')
    runner._load();model=runner._model;tok=runner._tokenizer
    model.load_adapter(str(folder),adapter_name='cyber',is_trainable=False,device_map={'':'cpu'})
    layers=[m for m in model.modules() if hasattr(m,'lora_A') and hasattr(m,'disable_adapters')]
    assert len(layers)==112
    all_features=[]; actual=[]
    for i,c in enumerate(old['cases']):
        prompt=old['prompt_template'].format(question=c['question'],**c['answers']); inputs=runner._inputs(prompt)
        assert inputs['input_ids'][0].tolist()==c['prompt_token_ids']
        with model.disable_adapter(),torch.inference_mode():
            assert all(m.disable_adapters for m in layers)
            torch.manual_seed(old['settings']['seed'])
            seq=model.generate(**inputs,max_new_tokens=old['settings']['max_new_tokens'],do_sample=old['settings']['do_sample'],use_cache=True,pad_token_id=tok.pad_token_id)
            gen=seq[0,inputs['input_ids'].shape[-1]:]
            text=tok.decode(gen,skip_special_tokens=True).strip()
            if text!=c['base']['text']: raise RuntimeError(f'Base reproduction mismatch case {i+1}: {text!r} vs {c["base"]["text"]!r}')
            non_eos=gen.tolist()
            while non_eos and non_eos[-1]==tok.eos_token_id: non_eos.pop()
            if not non_eos: raise RuntimeError('No answer token to probe')
            ids=torch.cat([inputs['input_ids'],torch.tensor([non_eos],device=inputs['input_ids'].device)],dim=1)
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),output_hidden_states=True,use_cache=False,return_dict=True)
            h=result.hidden_states[-1][0,-min(4,len(non_eos)):,:].detach().float().cpu()
            if h.shape[0]<4: h=torch.cat([h[:1].repeat(4-h.shape[0],1),h],0)
            all_features.append(h); actual.append(text)
        if (i+1)%10==0: print(f'Collected base answer and hidden states {i+1}/80',flush=True)
    X=torch.stack(all_features); Y=torch.tensor([float(c['base']['correct']) for c in old['cases']])
    torch.save({'features':X,'labels':Y,'spec':spec},features_path)
    fit=X[train]; mu=fit.mean((0,1),keepdim=True); sd=fit.std((0,1),keepdim=True).clamp_min(1e-6)
    standardized=(X-mu)/sd
    basis=svd_basis(standardized[train],8)
    projected=standardized@basis
    torch.manual_seed(42)
    lin=LinearStep(8,16,0.1);trf=TransformerStep(8,16,4,0.1,4)
    opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in (lin,trf)]
    for m,opt in zip((lin,trf),opts):
        m.train()
        for epoch in range(120):
            opt.zero_grad(); logits=m(projected[train],None).mean(dim=1)
            loss=F.binary_cross_entropy_with_logits(logits,Y[train]);loss.backward();opt.step()
        m.eval()
    torch.save({'linear_state':lin.state_dict(),'transformer_state':trf.state_dict(),'mean':mu,'std':sd,'svd_basis':basis,'spec':spec},ROOT/'results/cyber_router_v1_heads.pt')
    with torch.inference_mode():
        lp=torch.sigmoid(lin(projected[test],None).mean(dim=1));tp=torch.sigmoid(trf(projected[test],None).mean(dim=1));probs=(lp+tp)/2
    rec={'spec':spec,'created_at':datetime.now(timezone.utc).isoformat(),'complete':False,'base_model':BASE_MODEL_ID,'base_revision':BASE_REVISION,'cyber_adapter_sha256':whash,'test_cases':[],'feature_artifact_sha256':hashlib.sha256(features_path.read_bytes()).hexdigest()}
    for j,i in enumerate(test):
        c=old['cases'][i];prob=float(probs[j]);switched=prob<.5
        row={'index':i+1,'p_correct_score':prob,'linear_score':float(lp[j]),'transformer_score':float(tp[j]),'switched':switched,'base':c['base'],'cyber_reference':c['cyber']}
        if switched:
            # Discard all earlier generation caches: runner._generate starts with the entire prompt and no past_key_values.
            prompt=old['prompt_template'].format(question=c['question'],**c['answers']);inputs=runner._inputs(prompt)
            fresh=runner._generate(inputs,old['settings'],True,adapter_name='cyber')
            if fresh['text']!=c['cyber']['text']: raise RuntimeError(f'Fresh cyber output differs from reference case {i+1}')
            assert all(not m.disable_adapters and list(m.active_adapters)==['cyber'] for m in layers)
            row['fresh_cyber']=fresh
        row['router_correct']=c['cyber']['correct'] if switched else c['base']['correct']
        rec['test_cases'].append(row)
        out.write_text(json.dumps(rec,indent=2),encoding='utf-8')
        print(f'Test {j+1}/40 switched={switched} score={prob:.3f}',flush=True)
    t=rec['test_cases']
    rec['metrics']={'n':len(t),'base_correct':sum(x['base']['correct'] for x in t),'cyber_correct':sum(x['cyber_reference']['correct'] for x in t),'router_correct':sum(x['router_correct'] for x in t),'oracle_correct':sum(x['base']['correct'] or x['cyber_reference']['correct'] for x in t),'switches':sum(x['switched'] for x in t),'useful_switches':sum(x['switched'] and not x['base']['correct'] and x['cyber_reference']['correct'] for x in t),'harmful_switches':sum(x['switched'] and x['base']['correct'] and not x['cyber_reference']['correct'] for x in t),'probe_correct':sum((x['p_correct_score']>=.5)==x['base']['correct'] for x in t)}
    rec['complete']=True;rec['finished_at']=datetime.now(timezone.utc).isoformat();out.write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec['metrics']),flush=True)
if __name__=='__main__':main()
