"""Validate an untrained architecture and export; does not estimate accuracy."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def cpu_check(folder, model_name='architecture.onnx'):
    import ctypes
    from ctypes import wintypes
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder/model_name), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    results = []
    for fixture in sorted(folder.glob('case_*.npz')):
        with np.load(fixture) as data:
            inputs = {'frames': data['frames'], 'intervals': data['intervals']}
            reference = data['logits']
        predicted = session.run(['logits'], inputs)[0]
        assert predicted.shape == (5,) and predicted.dtype == np.float32
        assert np.isfinite(predicted).all()
        np.testing.assert_allclose(predicted, reference, rtol=1e-3, atol=1e-4)
        times = []
        for _ in range(20):
            start = perf_counter()
            session.run(['logits'], inputs)
            times.append(perf_counter()-start)
        results.append(dict(case=fixture.stem, mean_s=float(np.mean(times)),
                            p95_s=float(np.percentile(times, 95)),
                            max_abs_error=float(np.max(np.abs(predicted-reference)))))
    class Memory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    kernel = ctypes.WinDLL('kernel32')
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
        raise ctypes.WinError()
    report = dict(providers=session.get_providers(), threads=4, repetitions=20,
                  peak_working_set_bytes=memory.PeakWorkingSetSize,
                  peak_commit_bytes=memory.PeakPagefileUsage, cases=results)
    assert max(r['mean_s'] for r in results) < 0.7, report
    assert max(memory.PeakWorkingSetSize, memory.PeakPagefileUsage) < 6_000_000_000
    (folder/'cpu_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main(folder):
    import blosc2
    import onnx
    import torch
    from roomsense_model import RoomSenseModel, SubmissionModel, export_submission
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    def rows(path):
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    train = rows('analysis/splits/holdout_a1/train_windows.csv')[0]
    val = rows('analysis/splits/holdout_a1/val_windows.csv')[0]
    stats = rows('analysis/window_statistics.csv')
    gap = max((r for r in stats if r['split']=='labeled'), key=lambda r: float(r['ati_max_us']))
    trace = next(r for r in rows('analysis/trace_statistics.csv') if r['identifier']==gap['identifier'])
    gap = dict(gap, csi_path=trace['csi_path'], ati_path=trace['ati_path'])
    def load(row):
        start, stop = int(row['start']), int(row['stop'])
        csi = blosc2.load_array(row['csi_path'])
        frames = csi[start:stop].view(np.float32).reshape(600,1,3,1024,2).copy()
        intervals = blosc2.load_array(row['ati_path'])[start:stop].copy()
        intervals[-1] = 0
        return frames, intervals
    cases = [('train', load(train)), ('heldout', load(val)), ('gap', load(gap)),
             ('zero', (np.zeros((600,1,3,1024,2), np.float32), np.zeros(600,np.float32)))]
    model = RoomSenseModel()
    inputs = tuple(torch.from_numpy(a).unsqueeze(0) for a in cases[0][1])
    assert torch.cuda.is_available()
    model.cuda().train()
    gpu_inputs = tuple(x.cuda() for x in inputs)
    # Verify batched training and finite parameter gradients without fitting weights.
    batched = tuple(x.repeat(2, *([1]*(x.ndim-1))) for x in gpu_inputs)
    loss = torch.nn.functional.cross_entropy(model(*batched), torch.tensor([0,1],device='cuda'))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        gpu_logits = model(*gpu_inputs).cpu().numpy()
    model.cpu()
    with torch.no_grad():
        np.testing.assert_allclose(model(*inputs).numpy(), gpu_logits, rtol=1e-3, atol=1e-4)
        # Padding has no effect; magnitude is invariant to 90-degree phase rotation.
        changed = inputs[1].clone()
        changed[:,-1] = 999999
        torch.testing.assert_close(model(*inputs), model(inputs[0],changed))
        rotated = torch.stack((-inputs[0][...,1], inputs[0][...,0]), dim=-1)
        torch.testing.assert_close(model(*inputs), model(rotated,inputs[1]))
    submission = SubmissionModel(model).eval()
    for name, arrays in cases:
        with torch.no_grad():
            logits = submission(*(torch.from_numpy(a) for a in arrays)).numpy()
        assert np.isfinite(logits).all()
        np.savez(folder/f'case_{name}.npz', frames=arrays[0], intervals=arrays[1], logits=logits)
    example = tuple(torch.from_numpy(a) for a in cases[0][1])
    export_submission(model, example, folder/'architecture.onnx')
    onnx.checker.check_model(str(folder/'architecture.onnx'))
    size = (folder/'architecture.onnx').stat().st_size
    assert size < 100_000_000
    info = dict(parameters=sum(p.numel() for p in model.parameters()), onnx_bytes=size,
                trained=False, gpu_backward='pass', batch_size_tested=2,
                ignored_final_interval='pass', common_phase_rotation_invariance='pass',
                gpu_cpu_numerical_agreement='pass')
    (folder/'architecture_validation.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2), flush=True)
    subprocess.run([sys.executable,'-X','utf8',__file__,'--output',str(folder),'--cpu-only'],check=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/model_design'))
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--model-name',default='architecture.onnx')
    args=parser.parse_args()
    cpu_check(args.output,args.model_name) if args.cpu_only else main(args.output)
