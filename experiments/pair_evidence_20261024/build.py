"""Exploratory temporal debias consensus guard on exact scored .48 graph."""
import csv
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path
import blosc2
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/pair_evidence_20261024')
EXPORT=WORK/'export'
SOURCE=ROOT/'models/original_one_guard_20261024/01_original_one_guard.onnx'
SOURCE_HASH='021769f76948def0ec0d53cbe54d779e84beef925226a3e6498c526d18901324'
PREFIX='__temporal_rescue80_'
RAW=['__rxf_v0___tta_v0___roomsense_bias_0_before','__rxf_v0___tta_v1___roomsense_bias_0_before']
BASE=PREFIX+'base'
ELIGIBLE=lambda prediction: prediction==0


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def engine(model):
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    return ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])


def policy(base,raw0,raw1):
    trigger=int(base.argmax())==0 and all(int(z.argmax())==1 and float(z[1]-z[0])>=float(np.float32(np.log(4))) for z in (raw0,raw1))
    return (raw0 if trigger else base),trigger


def assemble():
    assert sha(SOURCE)==SOURCE_HASH
    source=onnx.load(SOURCE);m=onnx.ModelProto();m.CopyFrom(source)
    assert m.graph.node[-1].output[0]=='logits'
    assert not any('logits' in n.input for n in m.graph.node)
    assert all(any(name in n.output for n in m.graph.node) for name in RAW)
    m.graph.node[-1].output[0]=BASE
    for name,value in [('zero',np.array(0,np.int64)),('one',np.array(1,np.int64)),('margin',np.array(np.log(4),np.float32))]:
        m.graph.initializer.append(numpy_helper.from_array(value,PREFIX+name))
    nodes=[helper.make_node('ArgMax',[BASE],[PREFIX+'base_class'],axis=0,keepdims=0),
           helper.make_node('Equal',[PREFIX+'base_class',PREFIX+'zero'],[PREFIX+'base_empty'])]
    for k,raw in enumerate(RAW):
        p=PREFIX+str(k)+'_'
        nodes.extend([helper.make_node('ArgMax',[raw],[p+'class'],axis=0,keepdims=0),
            helper.make_node('Equal',[p+'class',PREFIX+'one'],[p+'is_one']),
            helper.make_node('Gather',[raw,PREFIX+'zero'],[p+'zero_score'],axis=0),
            helper.make_node('Gather',[raw,PREFIX+'one'],[p+'one_score'],axis=0),
            helper.make_node('Sub',[p+'one_score',p+'zero_score'],[p+'margin']),
            helper.make_node('GreaterOrEqual',[p+'margin',PREFIX+'margin'],[p+'confident']),
            helper.make_node('And',[p+'is_one',p+'confident'],[p+'supports'])])
    nodes.extend([helper.make_node('And',[PREFIX+'0_supports',PREFIX+'1_supports'],[PREFIX+'agree']),
        helper.make_node('And',[PREFIX+'agree',PREFIX+'base_empty'],[PREFIX+'trigger']),
        helper.make_node('Where',[PREFIX+'trigger',RAW[0],BASE],['logits'])])
    m.graph.node.extend(nodes)
    for old,new in zip(source.graph.node,m.graph.node):
        v=onnx.NodeProto();v.CopyFrom(new)
        if BASE in v.output:v.output[0]='logits'
        assert old.SerializeToString()==v.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(source.graph.initializer,m.graph.initializer))
    assert source.graph.input==m.graph.input and source.graph.output==m.graph.output
    props={x.key:x.value for x in source.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=SOURCE_HASH,
        transformation='Only when base048 argmax0, both original/time-reversed pre-bias views argmax1 and each p1/p0>=4: output original pre-bias logits; else exact base048 logits.',
        training='Original learned weights and fixed final bias unchanged; guard reads pre-bias evidence only.',
        caveat='Exploratory platform candidate. Source-data diagnostics do not predict hidden score; fixed .8 means binary conditional odds, not calibrated five-class confidence.')
    helper.set_model_props(m,props);onnx.checker.check_model(m)
    EXPORT.mkdir(parents=True,exist_ok=True);onnx.save_model(m,EXPORT/'model.onnx',save_as_external_data=False)
    save(EXPORT/'structure_proof.json',dict(base_sha256=SOURCE_HASH,candidate_sha256=sha(EXPORT/'model.onnx'),
        original_nodes=len(source.graph.node),added_nodes=len(nodes),original_initializers=len(source.graph.initializer),
        original_weights_and_body_byte_preserved=True,all_nonzero_base_predictions_structurally_unchanged=True,
        added_constants='int64 zero, int64 one, float32 log4',embedded=True))


def audit():
    # Extra graph outputs expose existing tensors without altering any node.
    m=onnx.load(EXPORT/'model.onnx')
    for name in [BASE,*RAW]:m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[5]))
    session=engine(m.SerializeToString());del m
    with (Path('D:/huawei2026-experiments/actual047_audit_20261024/all_predictions.csv')).open(encoding='utf-8-sig',newline='') as f:
        old=list(csv.DictReader(f))
    lookup={(r['identifier'],int(r['start'])):r for r in old}
    joined=read(Path('D:/huawei2026-experiments/remaining048_20261024/joined_windows.json'))
    by_record={}
    for row in joined:
        if ELIGIBLE(row['prediction']):by_record.setdefault(row['identifier'],[]).append(row)
    digest=sha(EXPORT/'model.onnx');checks=[];references=[];base_error=0.;rule_error=0.
    selected_examples=set()
    for identifier,rows in by_record.items():
        result_path=EXPORT/'records'/f'{identifier}.json'
        if result_path.exists():
            value=read(result_path);assert value['candidate_sha256']==digest
            checks.extend(value['windows']);continue
        raw=Path('D:/huawei2026-data/labeled')
        csi=blosc2.load_array(str(raw/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{identifier}.b2')).reshape(-1)
        local=[]
        for row in rows:
            start,stop=row['start'],row['stop']
            inputs=dict(frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2),
                        intervals=ati[start:stop].astype(np.float32,copy=True));inputs['intervals'][-1]=0
            out,base,raw0,raw1=session.run(None,inputs)
            expected,trigger=policy(base,raw0,raw1)
            assert out.dtype==np.float32 and np.isfinite(out).all() and np.array_equal(out,expected)
            cached=np.array([float(lookup[identifier,start][f'light047_logit_{k}']) for k in range(5)],np.float32)
            error=float(np.abs(cached-base).max());assert error<1e-5
            base_error=max(base_error,error)
            assert int(base.argmax())==row['prediction']
            local.append(dict(identifier=identifier,start=start,room=row['room'],truth=row['truth'],base=row['prediction'],
                prediction=int(out.argmax()),trigger=trigger,original_raw_prediction=int(raw0.argmax()),
                reverse_raw_prediction=int(raw1.argmax()),original_margin=float(raw0[1]-raw0[0]),reverse_margin=float(raw1[1]-raw1[0]),
                base_logit_error=error,rule_logit_error=0.))
            category=('rescued' if int(out.argmax())==row['truth'] else ('false_positive' if int(base.argmax())==row['truth'] else 'changed_wrong')) if trigger else 'unchanged'
            if category not in selected_examples:
                path=EXPORT/f'fixture_{category}.npz';np.savez_compressed(path,**inputs)
                references.append(dict(input=str(path),logits=out.tolist(),route=category));selected_examples.add(category)
        save(result_path,dict(candidate_sha256=digest,windows=local));checks.extend(local)
        print('AUDIT_RECORD',identifier,'windows',len(local),'trigger',sum(r['trigger'] for r in local),flush=True)
        del csi,ati;gc.collect()
    actual={(r['identifier'],r['start']):r for r in checks}
    combined=[]
    for row in joined:
        value=actual.get((row['identifier'],row['start']))
        combined.append(dict(**row,candidate_prediction=value['prediction'] if value else row['prediction'],
                             verification='raw_instrumented_same_graph' if value else 'exact_structural_nonzero_preservation'))
    assert len(checks)==sum(ELIGIBLE(r['prediction']) for r in joined)
    # Preserve active-branch fixtures when resuming already verified records.
    existing={r['input'] for r in references}
    for fixture in EXPORT.glob('fixture_*.npz'):
        if str(fixture) in existing:continue
        with np.load(fixture) as d:inputs={k:d[k].copy() for k in ('frames','intervals')}
        out,base,r0,r1=session.run(None,inputs);expected,trigger=policy(base,r0,r1)
        assert np.array_equal(out,expected)
        references.append(dict(input=str(fixture),logits=expected.tolist(),route='trigger' if trigger else 'base'))
    # Nonzero and edge fixtures check the original single-output exported file.
    for path in [ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero']:
        if path=='zero':inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
        else:
            with np.load(path) as d:inputs={k:d[k].copy() for k in ('frames','intervals')}
        out,base,r0,r1=session.run(None,inputs);expected,trigger=policy(base,r0,r1)
        assert np.array_equal(out,expected)
        references.append(dict(input=str(path),logits=expected.tolist(),route='trigger' if trigger else 'base'))
    del session;gc.collect()
    save(EXPORT/'references.json',references)
    cm=np.zeros((5,5),np.int64)
    for r in combined:cm[r['truth'],r['candidate_prediction']]+=1
    result=dict(candidate_sha256=digest,total=960,correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/960),
        corrected=sum(r['prediction']!=r['truth'] and r['candidate_prediction']==r['truth'] for r in combined),
        broken=sum(r['prediction']==r['truth'] and r['candidate_prediction']!=r['truth'] for r in combined),
        confusion_matrix=cm.tolist(),raw_directly_checked=len(checks),unchanged_by_structure=960-len(checks),
        base_logit_max_error=max(r['base_logit_error'] for r in checks),rule_logit_max_error=0.,
        independent_test=False,hidden_accuracy=None)
    save(EXPORT/'validation.json',result);save(EXPORT/'all_predictions.json',combined)
    print('VALIDATION',result,flush=True)


if __name__=='__main__':
    assemble();audit()
