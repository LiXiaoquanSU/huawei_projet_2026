"""Check canonical retained models against their recorded SHA256 and size."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',action='store_true',help='Also run the current best model on zero input')
    args=parser.parse_args()
    registry=json.loads((ROOT/'models/registry.json').read_text(encoding='utf-8'))
    checked=[]
    for item in registry['retained_models']:
        path=(ROOT/item['path']).resolve()
        assert path.is_relative_to(ROOT/'models') and path.is_file()
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert digest==item['sha256'] and path.stat().st_size==item['bytes']
        assert item['bytes']<100_000_000
        checked.append(dict(id=item['id'],score=item['platform_score'],sha256=digest))
    if args.runtime:
        import numpy as np
        import onnxruntime as ort
        best=ROOT/registry['current_best']['path']
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(best),sess_options=options,providers=['CPUExecutionProvider'])
        actual=[(v.name,v.shape,v.type) for v in session.get_inputs()]
        assert actual==[('frames',[600,1,3,1024,2],'tensor(float)'),('intervals',[600],'tensor(float)')]
        output=session.run(['logits'],dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32)))[0]
        assert output.shape==(5,) and output.dtype==np.float32 and np.isfinite(output).all()
    print(json.dumps(dict(status='pass',models=checked,best_score=registry['current_best']['platform_score'],runtime_checked=args.runtime),indent=2))


if __name__=='__main__':main()
