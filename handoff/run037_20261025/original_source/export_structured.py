"""Export a trained structured model, or clearly labelled untrained architecture."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import onnx
from structured_model import StructuredModel
from roomsense_model import SubmissionModel,export_submission


def export(model,folder,architecture=False):
    folder.mkdir(parents=True,exist_ok=True);model=model.cpu().eval();wrapper=SubmissionModel(model);first=None;diagnostics={}
    for path in Path('artifacts/model_design').glob('case_*.npz'):
        with np.load(path) as d:x,dt=d['frames'].copy(),d['intervals'].copy()
        a,b=torch.from_numpy(x),torch.from_numpy(dt)
        with torch.no_grad():
            z=wrapper(a,b).numpy();t,s=model.preprocess(a[None],b[None]);tasks=model.forward_tasks(t,s)
        np.savez(folder/path.name,frames=x,intervals=dt,logits=z)
        diagnostics[path.stem]={k:tasks[k].tolist() for k in ['direct','ordinal','cumulative','gate','signal']}
        if first is None:first=(a,b)
    filename='architecture.onnx' if architecture else 'model.onnx'
    export_submission(model,first,folder/filename)
    onnx.checker.check_model(str(folder/filename));assert (folder/filename).stat().st_size<100_000_000
    (folder/'subtask_diagnostics.json').write_text(json.dumps(dict(note='signal is a coherence proxy; context is NOT physical room area; agreement is NOT accuracy',mode=model.mode,size=model.size,untrained_architecture=architecture,ordinal_head_trained=not architecture and model.mode!='direct',signal_head_trained=not architecture and model.mode in ['auxiliary','physical'],cases=diagnostics),indent=2))
    subprocess.run([sys.executable,'-X','utf8','validate_model.py','--cpu-only','--output',str(folder),'--model-name',filename],check=True)
    print('EXPORTED',folder/filename,(folder/filename).stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--size',choices=['small','large','spectral'],default='large');p.add_argument('--mode',choices=['direct','ordinal','auxiliary','physical'],default='auxiliary');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    model=StructuredModel(args.size,args.mode)
    if args.checkpoint:model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model_state'])
    export(model,args.output,architecture=args.checkpoint is None)
