"""Export/check selected full-data research models without replacing scored models."""
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from net import ROOT, fresh, Submission
from run import WORK,read,save,sha

def export(variant,seed=42):
    key=f'{variant}_s{seed}_full';fit=WORK/'fits'/key
    state=torch.load(fit/'final.pt',map_location='cpu',weights_only=True)
    model=fresh('baseline' if variant=='external_transfer' else variant,seed)
    model.load_state_dict(state['model_state']);wrapper=Submission(model.eval()).eval()
    out=WORK/'exports'/key;out.mkdir(parents=True,exist_ok=True)
    references=[];example=None
    for file in [ROOT/'checks/smoke_input.npz',ROOT/'_archive/history_20261019/artifacts/model_design/case_gap.npz','zero']:
        if file=='zero':x,dt=torch.zeros(600,1,3,1024,2),torch.zeros(600)
        else:
            with np.load(file) as d:x,dt=[torch.from_numpy(d[k].copy()) for k in ['frames','intervals']]
        with torch.no_grad():z=wrapper(x,dt)
        assert torch.isfinite(z).all();references.append(dict(input=str(file),logits=z.tolist()))
        if example is None:example=(x,dt)
    save(out/'references.json',references)
    torch.onnx.export(wrapper,example,str(out/'model.onnx'),input_names=['frames','intervals'],output_names=['logits'],
                      opset_version=18,dynamo=False,external_data=False)
    checker=ROOT/'tools/check_candidate.py'
    subprocess.run([sys.executable,'-X','utf8',str(checker),str(out)],check=True)
    resource=read(out/'resource_check.json')
    target=ROOT/'models/motion_external_20261021'/f'{variant}_seed{seed}.onnx'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==resource['sha256']
    else:shutil.copy2(out/'model.onnx',target)
    metadata=dict(variant=variant,path=str(target.relative_to(ROOT)).replace('\\','/'),
                  sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,
                  status='research_candidate_platform_unknown',resource=resource,
                  external_data=variant=='external_transfer',checkpoint=str(fit/'final.pt'),
                  config=state['config'],training_records=48,
                  note='Full48 fit has no independent local accuracy; development results belong to separate fold fits.')
    if variant=='external_transfer':metadata['submission_status']='Research only; external-data competition permission/model distribution conditions unverified.'
    save(target.with_suffix('.json'),metadata)
    print('EXPORTED',metadata['path'],metadata['bytes'],flush=True)
    del model,wrapper,state;gc.collect()
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',nargs='*');args=p.parse_args()
    torch.set_num_threads(4)
    variants=args.variant or [r['variant'] for r in read(WORK/'selection_stage1.json')['selected']]
    results=[export(v) for v in variants]
    save(WORK/'exports'/('external_registry.json' if variants==['external_transfer'] else 'architecture_registry.json'),results)
