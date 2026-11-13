"""Export selected fixed policy using two preprocesses and batched frozen features."""
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import blosc2
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort
import torch
from common import ROOT,WORK,Submission,build,rows,read,save,sha


def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def probability(z):
    p=np.exp(np.asarray(z,dtype=float)-np.max(z));return p/p.sum()


def compact(source,dest):
    model=onnx.load(source);casts=[];arrays=[]
    for value in model.graph.initializer:
        if value.data_type==TensorProto.FLOAT and np.prod(value.dims)>=1024:
            array=numpy_helper.to_array(value);half=array.astype(np.float16);assert np.isfinite(half).all()
            # Source inference weights were already rounded identically to .44.
            assert np.array_equal(array,half.astype(np.float32))
            name=value.name+'_storage16';arrays.append(numpy_helper.from_array(half,name))
            casts.append(helper.make_node('Cast',[name],[value.name],to=TensorProto.FLOAT))
        else:arrays.append(value)
    nodes=list(model.graph.node);del model.graph.node[:];model.graph.node.extend(casts+nodes)
    del model.graph.initializer[:];model.graph.initializer.extend(arrays)
    onnx.checker.check_model(model);onnx.save_model(model,dest,save_as_external_data=False)


def main():
    torch.set_num_threads(4);assert read(WORK/'complete.json')['status']=='pass'
    selection=read(WORK/'selection.json');policy=selection['policy']
    protected={p:sha(ROOT/p) for p in ('models/best_044/model.onnx','models/best_047_light/model.onnx','models/best_047_full/model.onnx')}
    model,bias=build(ROOT/'models/best_037/final.pt',full=True);model=model.cuda()
    candidate=Submission(model,bias,policy).cuda().eval()
    baselines={name:Submission(model,bias,name).cuda().eval() for name in ('light','full')}
    originals={name:session(ROOT/f'models/best_047_{name}/model.onnx') for name in ('light','full')}
    cases=[];references=[]
    def probe(x,dt,name,fixture=False):
        tx=torch.from_numpy(x).cuda();td=torch.from_numpy(dt).cuda();checks={}
        with torch.no_grad():
            for variant in ('light','full'):
                actual=originals[variant].run(['logits'],dict(frames=x,intervals=dt))[0]
                expected=baselines[variant](tx,td).cpu().numpy()
                error=float(np.abs(probability(expected)-probability(actual)).max())
                margin=float(np.sort(probability(actual))[-1]-np.sort(probability(actual))[-2])
                same=int(expected.argmax())==int(actual.argmax())
                assert error<.001 and (same or margin<.001),(name,variant,error)
                checks[variant]=dict(probability_max_error=error,argmax_matches=same)
            output=candidate(tx,td).cpu().numpy()
        assert output.shape==(5,) and np.isfinite(output).all()
        cases.append(dict(input=str(name),baseline_equivalence=checks,candidate_prediction=int(output.argmax())))
        if fixture:references.append(dict(input=str(name),logits=output.tolist()))
    for path in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero'):
        if path=='zero':x,dt=np.zeros((600,1,3,1024,2),np.float32),np.zeros(600,np.float32)
        else:
            with np.load(path) as data:x,dt=[data[k].copy() for k in ('frames','intervals')]
        probe(x,dt,path,True)
        if len(references)==1:example=tuple(torch.from_numpy(v.copy()) for v in (x,dt))
    raw=Path('D:/huawei2026-data/labeled');chosen=[];rooms_seen=set();labels_seen=set()
    for row in rows(raw/'samples.csv'):
        if row['room'] not in rooms_seen or row['headcount'] not in labels_seen:
            chosen.append(row);rooms_seen.add(row['room']);labels_seen.add(row['headcount'])
    for row in chosen:
        csi=blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"))
        for start in (0,5700,11400):
            x=csi[start:start+600].view(np.float32).reshape(600,1,3,1024,2).copy();dt=ati[start:start+600].copy();dt[-1]=0
            probe(x,dt,f"{row['identifier']}:{start}")
        del csi,ati
        print('BASELINE_EQUIVALENCE',row['identifier'],flush=True)
    del originals,baselines;gc.collect();candidate=candidate.cpu().eval();torch.cuda.empty_cache()
    export=WORK/'export';export.mkdir(exist_ok=True);save(export/'references.json',references)
    torch.onnx.export(candidate,example,str(export/'full_precision.onnx'),input_names=['frames','intervals'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
    compact(export/'full_precision.onnx',export/'model.onnx')
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(export)],check=True)
    resource=read(export/'resource_check.json')
    # Compare exported model to the Torch candidate on every original 960 anchor.
    # Feature-only evaluation is not a raw ONNX check, so do not claim that here.
    audit=dict(status='pass',cases=cases,description='batched feature permutation baseline reproduces actual two .47 files; numerical probes only, not independent accuracy')
    save(WORK/'baseline_equivalence.json',audit)
    target=ROOT/'models/tta_policy_20261024'/f'01_symmetric_{policy}.onnx';target.parent.mkdir(exist_ok=True)
    assert not target.exists();shutil.copy2(export/'model.onnx',target)
    meta=dict(path=target.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='policy_research_platform_unverified',
              method='same .44 weights and bias; two raw preprocesses, feature-space receiver permutations, batched neural evaluation; selected fixed policy '+policy,
              policy=policy,selection=selection,stress=read(WORK/'stress.json'),resource=resource,numerical_audit=audit,
              parameters_changed=False,new_training_fits=0,report='docs/TTA_POLICY_20261024.md')
    save(target.with_suffix('.json'),meta);save(WORK/'candidate.json',meta)
    for path,h in protected.items():assert sha(ROOT/path)==h
    source_files=[*list(Path(__file__).parent.glob('*.py')),*list((ROOT/'src').glob('*.py')),ROOT/'tools/check_candidate.py']
    manifest=[]
    for path in source_files:
        relative=path.relative_to(ROOT);dest=WORK/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        manifest.append(dict(path=relative.as_posix(),sha256=sha(path)))
    save(WORK/'source_manifest.json',manifest)
    print('DELIVERED',json.dumps({k:meta[k] for k in ('path','bytes','sha256','policy')}),flush=True)


if __name__=='__main__':main()
