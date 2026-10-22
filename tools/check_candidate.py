"""Isolated ONNX Runtime CPU check; deliberately imports neither torch nor CUDA."""
import argparse,ctypes,hashlib,json,time
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort

def peak_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    c=Counters();c.cb=ctypes.sizeof(c)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
    psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise ctypes.WinError()
    return dict(peak_working_set_bytes=c.PeakWorkingSetSize,peak_commit_bytes=c.PeakPagefileUsage)

def probs(x):
    p=np.exp(x-x.max());return p/p.sum()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();path=a.folder/'model.onnx'
    graph=onnx.load(path);onnx.checker.check_model(graph)
    external=[t.name for t in graph.graph.initializer if t.data_location==onnx.TensorProto.EXTERNAL];assert not external
    del graph
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    assert [(v.name,v.shape,v.type) for v in session.get_inputs()]==[('frames',[600,1,3,1024,2],'tensor(float)'),('intervals',[600],'tensor(float)')]
    cases=json.loads((a.folder/'references.json').read_text());results=[]
    for case in cases:
        if case['input']=='zero':inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
        else:
            with np.load(case['input']) as d:inputs={k:d[k].copy() for k in ['frames','intervals']}
        reference=np.asarray(case['logits']);output=session.run(['logits'],inputs)[0]
        assert output.shape==(5,) and output.dtype==np.float32 and np.isfinite(output).all()
        pe=float(np.max(np.abs(probs(output)-probs(reference))));le=float(np.max(np.abs(output-reference)))
        same=int(output.argmax())==int(reference.argmax());margin=float(np.sort(probs(reference))[-1]-np.sort(probs(reference))[-2])
        assert pe<.002 and le<.02 and (same or margin<.002),(case,pe,le,same)
        elapsed=[]
        for _ in range(20):
            start=time.perf_counter();session.run(['logits'],inputs);elapsed.append(time.perf_counter()-start)
        results.append(dict(input=case['input'],probability_max_error=pe,logit_max_error=le,argmax_matches=same,reference_margin=margin,cpu_mean_s=float(np.mean(elapsed)),cpu_p95_s=float(np.percentile(elapsed,95))))
    memory=peak_memory();size=path.stat().st_size
    assert size<100_000_000 and max(r['cpu_mean_s'] for r in results)<.7 and max(memory.values())<6_000_000_000
    report=dict(status='pass',bytes=size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),embedded_weights=True,cpu_threads=4,cases=results,**memory,note='local CPU resource/numerical verification; not hidden platform accuracy')
    (a.folder/'resource_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
