"""Local file/runtime check; cannot diagnose a remote upload download failure."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--sample',type=Path,default=ROOT/'checks/smoke_input.npz');args=p.parse_args()
    model=onnx.load(args.model);onnx.checker.check_model(model)
    external=[t.name for t in model.graph.initializer if t.data_location==onnx.TensorProto.EXTERNAL]
    assert not external;assert args.model.stat().st_size<100_000_000
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.model),sess_options=options,providers=['CPUExecutionProvider'])
    if args.sample.exists():
        with np.load(args.sample) as d:inputs={k:d[k] for k in ['frames','intervals']}
        input_source=str(args.sample)
    else:
        inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32));input_source='synthetic zero input; interface check only'
    logits=session.run(['logits'],inputs)[0];assert logits.shape==(5,) and logits.dtype==np.float32 and np.isfinite(logits).all()
    times=[]
    for _ in range(5):
        start=time.perf_counter();session.run(['logits'],inputs);times.append(time.perf_counter()-start)
    print(json.dumps(dict(bytes=args.model.stat().st_size,sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),external_weights=external,onnx_valid=True,cpu_mean_s=float(np.mean(times)),input_source=input_source,logits=logits.tolist(),note='local smoke check only; not platform accuracy or proof of remote download access'),indent=2))
