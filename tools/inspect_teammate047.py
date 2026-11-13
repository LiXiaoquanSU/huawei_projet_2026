"""Analyze supplied scored artifacts without fitting parameters or reporting probe accuracy."""
import collections
import argparse
import csv
import gc
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import blosc2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
INBOX=Path('C:/Users/lxq_1/xwechat_files/wxid_6ct208ti8tam12_cc30/msg/file/2026-10')
SPECS=[('light','model(1).onnx','d41c3d7cff9a31cde80635ea40df30b4f432c9029b50b40430fff7af00ee62ad'),
       ('full','model(3).onnx','3e0da273c26ad3f966b2610a6bca351c365e111487dcbe6625dc30f90160fc33')]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,data):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def probability(x):
    p=np.exp(np.asarray(x,dtype=float)-np.max(x));return p/p.sum()


def structural(path,base_initializers,base_nodes):
    model=onnx.load(path);onnx.checker.check_model(model)
    tensors={x.name:x for x in model.graph.initializer}
    unchanged=[k for k in base_initializers if k in tensors and base_initializers[k].SerializeToString()==tensors[k].SerializeToString()]
    assert len(unchanged)==len(base_initializers)==178
    extras={}
    for name in tensors.keys()-base_initializers.keys():
        a=numpy_helper.to_array(tensors[name]);extras[name]=dict(shape=list(a.shape),dtype=str(a.dtype),
            values=a.tolist() if a.size<20 else dict(first=a[:5].tolist(),last=a[-5:].tolist()))
    # Prove each copied network body, including Constant attributes, is unchanged.
    first=base_nodes[0];groups=[]
    starts=[n for n in model.graph.node if n.op_type==first.op_type and n.name.endswith(first.name)]
    for start in starts:
        names=start.name[:-len(first.name)];values=start.output[0][:-len(first.output[0])]
        block=[n for n in model.graph.node if n.name.startswith(names)]
        assert len(block)==len(base_nodes)
        aliases={}
        for old,new in zip(base_nodes,block):
            for a,b in zip(old.input,new.input):
                if a in ('frames','intervals'):aliases[b]=a
        for old,new in zip(base_nodes,block):
            node=onnx.NodeProto();node.CopyFrom(new);node.name=node.name[len(names):]
            for i,value in enumerate(node.input):node.input[i]=aliases.get(value,value[len(values):] if value.startswith(values) else value)
            for i,value in enumerate(node.output):node.output[i]=value[len(values):] if value.startswith(values) else value
            assert node.SerializeToString()==old.SerializeToString()
        groups.append(dict(namespace=names,nodes=len(block),exact_body=True))
    result=dict(nodes=len(model.graph.node),initializers=len(tensors),old044_initializers_identical=len(unchanged),
                exact_network_body_copies=len(groups),body_proof=groups,
                additional_initializers=extras,op_counts=dict(collections.Counter(n.op_type for n in model.graph.node)),
                embedded_metadata={x.key:x.value for x in model.metadata_props},
                note='Embedded score/submission metadata is provenance text, not independent platform verification. Latest user-reported .47 overrides stale not-yet-evaluated text.')
    return result


def resource_checks(metadata,references,summary):
    for meta,(variant,_,_) in zip(metadata,SPECS):
        with tempfile.TemporaryDirectory(dir=ROOT/'checks',prefix='resource047_') as folder:
            folder=Path(folder)
            # A normal temporary copy also works under Windows restrictions on hard links.
            shutil.copyfile(ROOT/meta['path'],folder/'model.onnx')
            save(folder/'references.json',references[variant])
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/check_candidate.py'),str(folder)],check=True)
            resource=json.loads((folder/'resource_check.json').read_text());meta['resource']=resource
            save(ROOT/f'checks/teammate047_{variant}_resource.json',resource)
        meta['functional_audit']=summary
        save((ROOT/meta['path']).with_suffix('.json'),meta)
    analysis=json.loads((ROOT/'checks/teammate047_analysis.json').read_text())
    analysis['models']=metadata;analysis['resource_status']='pass'
    save(ROOT/'checks/teammate047_analysis.json',analysis)
    print('COMPLETE',json.dumps(summary),flush=True)


def resume_resources():
    analysis=json.loads((ROOT/'checks/teammate047_analysis.json').read_text())
    metadata=analysis['models']
    for meta in metadata:assert sha(ROOT/meta['path'])==meta['sha256']
    base=session(ROOT/'models/best_044/model.onnx');references={'light':[],'full':[]}
    order=np.r_[np.arange(598,-1,-1),599]
    for path in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero'):
        if path=='zero':x,dt=np.zeros((600,1,3,1024,2),np.float32),np.zeros(600,np.float32)
        else:
            with np.load(path) as data:x,dt=[data[k].copy() for k in ('frames','intervals')]
        views={}
        for permutation in itertools.permutations(range(3)):
            v=x[:,:,list(permutation),:,:].copy()
            a=base.run(['logits'],dict(frames=v,intervals=dt))[0]
            b=base.run(['logits'],dict(frames=v[::-1].copy(),intervals=dt[order].copy()))[0]
            views[permutation]=np.logaddexp(a.astype(float),b.astype(float))-np.log(2)
        identity=views[(0,1,2)]
        small=identity if identity.argmax() in (0,2) else np.logaddexp(identity,views[(2,1,0)])-np.log(2)
        large=small if small.argmax() in (0,2) else np.logaddexp.reduce(np.stack(list(views.values())),axis=0)-np.log(6)
        for key,value in [('light',small),('full',large)]:references[key].append(dict(input=str(path),logits=value.tolist()))
    del base;gc.collect()
    resource_checks(metadata,references,analysis['summary'])


def main():
    baseline=ROOT/'models/best_044/model.onnx';assert sha(baseline)=='d2c21c3161b3838247d73b4a23d9ba6091090be39a55627683a41604d04d9063'
    base_model=onnx.load(baseline);base_initializers={x.name:x for x in base_model.graph.initializer}
    metadata=[]
    for variant,filename,expected in SPECS:
        dest=ROOT/f'models/best_047_{variant}/model.onnx';dest.parent.mkdir(exist_ok=True)
        source=INBOX/filename
        if not source.exists():source=dest
        assert sha(source)==expected
        if dest.exists():assert sha(dest)==expected
        else:shutil.copy2(source,dest)
        meta=dict(id='best_047_'+variant,name='队友0.47 '+('4视角保护版' if variant=='light' else '12视角保护版'),
                  path=dest.relative_to(ROOT).as_posix(),sha256=expected,bytes=dest.stat().st_size,platform_score=.47,
                  score_source='user-reported platform result 2026-10-24',owner='teammate',submission_id=None,
                  original_filename=filename,status='tied_current_best',structure=structural(dest,base_initializers,base_model.graph.node),
                  report='docs/MODELS_047.md')
        metadata.append(meta)
    del base_model,base_initializers;gc.collect()
    base=session(baseline);light=session(ROOT/metadata[0]['path']);full=session(ROOT/metadata[1]['path'])
    permutations=list(itertools.permutations(range(3)));frame_order=np.arange(599,-1,-1);interval_order=np.r_[np.arange(598,-1,-1),599]
    cases=[];references={'light':[],'full':[]}

    def evaluate(x,dt,name,fixture=False):
        views={};per_view_predictions={}
        for permutation in permutations:
            v=x[:,:,list(permutation),:,:].copy()
            forward=base.run(['logits'],dict(frames=v,intervals=dt))[0]
            backward=base.run(['logits'],dict(frames=v[frame_order].copy(),intervals=dt[interval_order].copy()))[0]
            averaged=np.logaddexp(forward.astype(float),backward.astype(float))-np.log(2)
            views[permutation]=averaged
            per_view_predictions[''.join(map(str,permutation))]=[int(forward.argmax()),int(backward.argmax())]
        identity=views[(0,1,2)];reverse=views[(2,1,0)]
        expected_light=identity if int(identity.argmax()) in (0,2) else np.logaddexp(identity,reverse)-np.log(2)
        all_views=np.logaddexp.reduce(np.stack(list(views.values())),axis=0)-np.log(6)
        expected_full=expected_light if int(expected_light.argmax()) in (0,2) else all_views
        actual_light=light.run(['logits'],dict(frames=x,intervals=dt))[0]
        actual_full=full.run(['logits'],dict(frames=x,intervals=dt))[0]
        metrics={}
        for variant,expected,actual in [('light',expected_light,actual_light),('full',expected_full,actual_full)]:
            pe=float(np.abs(probability(expected)-probability(actual)).max());le=float(np.abs(expected-actual).max())
            same=int(expected.argmax())==int(actual.argmax());margin=float(np.sort(probability(expected))[-1]-np.sort(probability(expected))[-2])
            assert pe<.001 and le<.005 and (same or margin<.001),(name,variant,pe,le)
            metrics[variant]=dict(probability_max_error=pe,logit_max_error=le,argmax_matches=same)
            if fixture:references[variant].append(dict(input=str(name),logits=expected.tolist()))
        # This probes graph equivalence and decision changes, not correctness.
        cases.append(dict(input=str(name),time_average_prediction=int(identity.argmax()),
                          light_prediction=int(actual_light.argmax()),full_prediction=int(actual_full.argmax()),
                          light_gate_protected=int(identity.argmax()) in (0,2),full_gate_protected=int(expected_light.argmax()) in (0,2),
                          per_view_predictions=per_view_predictions,equivalence=metrics))
        print('PROBE',str(name),int(actual_light.argmax()),int(actual_full.argmax()),flush=True)

    for path in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero'):
        if path=='zero':x,dt=np.zeros((600,1,3,1024,2),np.float32),np.zeros(600,np.float32)
        else:
            with np.load(path) as d:x,dt=[d[k].copy() for k in ('frames','intervals')]
        evaluate(x,dt,path,fixture=True)
    raw=Path('D:/huawei2026-data/labeled');chosen=[];rooms_seen=set();labels_seen=set()
    for row in rows(raw/'samples.csv'):
        if row['room'] not in rooms_seen or row['headcount'] not in labels_seen:
            chosen.append(row);rooms_seen.add(row['room']);labels_seen.add(row['headcount'])
    for row in chosen:
        csi=blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"))
        for start in (0,5700,11400):
            x=csi[start:start+600].view(np.float32).reshape(600,1,3,1024,2).copy();dt=ati[start:start+600].copy();dt[-1]=0
            evaluate(x,dt,f"{row['identifier']}:{start}")
        del csi,ati
    del base,light,full;gc.collect()
    summary=dict(cases=len(cases),raw_windows=len(cases)-3,rooms=sorted(rooms_seen),labels=sorted(labels_seen),
                 light_full_prediction_disagreements=sum(c['light_prediction']!=c['full_prediction'] for c in cases),
                 light_protected_cases=sum(c['light_gate_protected'] for c in cases),
                 full_protected_cases=sum(c['full_gate_protected'] for c in cases),
                 max_probability_error=max(c['equivalence'][v]['probability_max_error'] for c in cases for v in ('light','full')),
                 note='Only functional/numerical probes. These source records were used for original training; no probe accuracy is claimed.')
    save(ROOT/'checks/teammate047_analysis.json',dict(date='2026-10-24',status='pass',summary=summary,cases=cases,models=metadata))
    # Each resource check runs in its own process, without the other sessions.
    resource_checks(metadata,references,summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-resources',action='store_true');args=parser.parse_args()
    if args.resume_resources:resume_resources()
    else:main()
