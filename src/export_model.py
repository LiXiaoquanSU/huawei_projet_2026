"""Export saved weights to a new ONNX file; leave the scored binary unchanged."""
import argparse
from pathlib import Path
import numpy as np,torch,onnx
from model import StructuredModel
from features import export_submission
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep existing models; choose a new filename')
    torch.set_num_threads(4);model=StructuredModel('spectral','physical')
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model_state'])
    fixture=ROOT/'checks/smoke_input.npz'
    if fixture.exists():
        with np.load(fixture) as d:example=tuple(torch.from_numpy(d[k].copy()) for k in ['frames','intervals'])
    else:example=(torch.zeros(600,1,3,1024,2),torch.zeros(600))
    args.output.parent.mkdir(parents=True,exist_ok=True);export_submission(model,example,args.output);onnx.checker.check_model(str(args.output))
    assert args.output.stat().st_size<100_000_000
    print(args.output.resolve());print('Run src/check_model.py on this file; platform accuracy remains unknown.')
