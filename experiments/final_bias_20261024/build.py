"""Append exactly one residual Add after frozen scored048; audit raw inputs."""
import csv
import gc
import hashlib
import json
from pathlib import Path
import blosc2
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/final_bias_20261024')
EXPORT=WORK/'export'
SOURCE=ROOT/'models/original_one_guard_20261024/01_original_one_guard.onnx'
SOURCE_HASH='021769f76948def0ec0d53cbe54d779e84beef925226a3e6498c526d18901324'
BASE='__final_bias_frozen048'
MAX_SHIFT=.350001


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def engine(value):
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    return ort.InferenceSession(value,sess_options=options,providers=['CPUExecutionProvider'])


def main():
    assert sha(SOURCE)==SOURCE_HASH
    summary=read(WORK/'summary.json');delta=np.asarray(summary['chosen_delta'],np.float32)
    assert delta.shape==(5,) and np.count_nonzero(delta)==1 and 0<delta[1]<=MAX_SHIFT
    old=onnx.load(SOURCE);model=onnx.ModelProto();model.CopyFrom(old)
    assert model.graph.node[-1].output[0]=='logits' and not any('logits' in n.input for n in model.graph.node)
    model.graph.node[-1].output[0]=BASE
    model.graph.initializer.append(numpy_helper.from_array(delta,'__final_bias_delta'))
    model.graph.node.append(helper.make_node('Add',[BASE,'__final_bias_delta'],['logits'],name='__final_bias_add'))
    for a,b in zip(old.graph.node,model.graph.node):
        n=onnx.NodeProto();n.CopyFrom(b)
        if BASE in n.output:n.output[0]='logits'
        assert a.SerializeToString()==n.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(old.graph.initializer,model.graph.initializer))
    assert old.graph.input==model.graph.input and old.graph.output==model.graph.output
    props={p.key:p.value for p in old.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=SOURCE_HASH,base_platform_accuracy='0.48',
        transformation=summary.get('transformation','After frozen048 final output, Add residual [0,0.35,0,0,0]. Original internal bias and routing untouched.'),
        training=summary.get('provenance','No backbone training. Bounded training-side whole-record calibration median; overlapping calibration subsets, not independent crossfits.'),
        caveat='Exploratory output-bias trial; target prior unknown and pure label shift unverified. May increase empty false positives.')
    helper.set_model_props(model,props);onnx.checker.check_model(model)
    EXPORT.mkdir(parents=True,exist_ok=True);onnx.save_model(model,EXPORT/'model.onnx',save_as_external_data=False)
    digest=sha(EXPORT/'model.onnx')
    save(EXPORT/'structure_proof.json',dict(base_sha256=SOURCE_HASH,candidate_sha256=digest,delta=delta.tolist(),
        original_nodes=len(old.graph.node),added_nodes=1,added_initializer='five float32 residual values',
        original_body_and_initializers_byte_preserved=True,original_internal_bias_and_routing_preserved=True,
        relative_scores_between0234_unchanged=True,no_new_backbone_inference=True))
    del old;gc.collect()
    rows=read(WORK/'rows.json')
    with np.load(WORK/'cached_exact048.npz') as values:
        logits=values['logits'].copy();labels=values['labels'].copy()
    expected=logits+delta;base_pred=logits.argmax(1);pred=expected.argmax(1)
    changed=set(np.flatnonzero(pred!=base_pred).tolist())
    margin=np.sort(expected,axis=1)[:,-1]-np.sort(expected,axis=1)[:,-2]
    selected=set(changed)|set(np.argsort(margin)[:12].tolist())
    for label in range(5):selected.add(int(np.flatnonzero(labels==label)[0]))
    by_record={}
    for i in sorted(selected):by_record.setdefault(rows[i]['identifier'],[]).append(i)
    model.graph.output.append(helper.make_tensor_value_info(BASE,TensorProto.FLOAT,[5]))
    session=engine(model.SerializeToString());del model;gc.collect()
    sources={r['identifier']:r for r in read(Path('D:/huawei2026-experiments/actual047_audit_20261024/summary.json'))['source_files']}
    checks=[];fixtures={};verified=[]
    for identifier,ix in by_record.items():
        raw=Path('D:/huawei2026-data/labeled')
        cpath=raw/f'CSI_{identifier}.b2';apath=raw/f'ATI_{identifier}.b2'
        assert sha(cpath)==sources[identifier]['csi_sha256'] and sha(apath)==sources[identifier]['ati_sha256']
        verified.append(sources[identifier])
        csi=blosc2.load_array(str(cpath));ati=blosc2.load_array(str(apath)).reshape(-1)
        for i in ix:
            r=rows[i];start,stop=r['start'],r['stop']
            inputs=dict(frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2),
                intervals=ati[start:stop].astype(np.float32,copy=True));inputs['intervals'][-1]=0
            out,base=session.run(None,inputs)
            be=float(np.abs(base-logits[i]).max());oe=float(np.abs(out-expected[i]).max())
            assert be<1e-5 and oe<1e-5 and int(out.argmax())==int(pred[i])
            checks.append(dict(identifier=identifier,start=start,truth=int(labels[i]),baseline=int(base_pred[i]),
                prediction=int(pred[i]),base_cache_error=be,output_reference_error=oe))
            category='corrected' if base_pred[i]!=labels[i] and pred[i]==labels[i] else ('broken' if base_pred[i]==labels[i] and pred[i]!=labels[i] else 'other')
            if pred[i]!=base_pred[i] and base_pred[i]>=2:category='nonzero_to_one'
            if category not in fixtures:
                path=EXPORT/f'fixture_{category}.npz';np.savez_compressed(path,**inputs);fixtures[category]=str(path)
        del csi,ati;gc.collect()
        print('RAW_BIAS_CHECK',identifier,len(ix),flush=True)
    del session;gc.collect()
    # Independent reference: actual original048 uninstrumented output + NumPy Add.
    session=engine(str(SOURCE));refs=[]
    for path in [*fixtures.values(),str(ROOT/'checks/smoke_input.npz'),str(ROOT/'checks/case_gap.npz'),'zero']:
        if path=='zero':inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
        else:
            with np.load(path) as d:inputs={k:d[k].copy() for k in ('frames','intervals')}
        z=session.run(['logits'],inputs)[0]
        refs.append(dict(input=path,logits=(z+delta).tolist(),reference='Independent original048 file + NumPy float32 residual Add'))
    del session;gc.collect()
    cm=np.bincount(labels*5+pred,minlength=25).reshape(5,5)
    validation=dict(candidate_sha256=digest,correct=int((pred==labels).sum()),total=len(pred),accuracy=float((pred==labels).mean()),
        corrected=int(((base_pred!=labels)&(pred==labels)).sum()),broken=int(((base_pred==labels)&(pred!=labels)).sum()),
        confusion_matrix=cm.tolist(),all960_method='Derive from previously actual-ONNX-verified048 logits using exact float32 Add; not960 new raw calls.',
        directly_raw_checked=len(checks),all_changed_windows_directly_checked=len(changed),
        base_logit_max_error=max(r['base_cache_error'] for r in checks),output_logit_max_error=max(r['output_reference_error'] for r in checks),
        independent_test=False,platform_score=None)
    save(EXPORT/'validation.json',validation);save(EXPORT/'references.json',refs)
    save(EXPORT/'raw_checks.json',checks);save(EXPORT/'source_proof.json',verified)
    print('FINAL_BIAS_VALIDATION',validation,flush=True)


if __name__=='__main__':main()
