"""Smoke-test GPU training followed by CPU-only ONNX inference, not model quality."""

import argparse
import csv
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from example import ExampleModel, load_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gpu_smoke.onnx"))
    args = parser.parse_args()
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU verification cannot proceed")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(42)
    convolution = torch.nn.Conv1d(3, 8, 3, padding=1).cuda()
    convolution(torch.randn(2, 3, 600, device="cuda")).square().mean().backward()
    torch.cuda.synchronize()
    print("GPU convolution forward/backward: PASS", flush=True)
    with (args.data_dir / "samples.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    csi, ati = load_sample(args.data_dir, row["identifier"])
    frames = csi[:600].view(np.float32).reshape(600, 1, 3, 1024, 2).copy()
    intervals = ati[:600].copy()
    intervals[-1] = 0
    inputs_cpu = (torch.from_numpy(frames), torch.from_numpy(intervals))
    inputs_gpu = tuple(x.cuda() for x in inputs_cpu)
    model = ExampleModel().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = torch.tensor([int(row["headcount"])], device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(*inputs_gpu).unsqueeze(0), target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    with torch.no_grad():
        result_gpu = model.eval()(*inputs_gpu).cpu().numpy()
    print("GPU forward/backward/optimizer: PASS", flush=True)
    model = model.cpu().eval()
    with torch.no_grad():
        result_cpu = model(*inputs_cpu).numpy()
    np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-4, atol=1e-3)
    torch.onnx.export(
        model, inputs_cpu, str(args.output),
        input_names=["frames", "intervals"], output_names=["logits"],
        opset_version=18, external_data=False,
    )
    onnx.checker.check_model(str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs = {"frames": frames, "intervals": intervals}
    result = session.run(["logits"], inputs)[0]
    assert result.shape == (5,) and result.dtype == np.float32
    assert np.isfinite(result).all()
    assert args.output.stat().st_size < 100_000_000
    np.testing.assert_allclose(result, result_cpu, rtol=1e-4, atol=1e-3)
    from time import perf_counter
    start = perf_counter()
    for _ in range(20):
        session.run(["logits"], inputs)
    elapsed = (perf_counter() - start) / 20
    print(f"CPU ONNX inference: PASS; mean {elapsed:.4f} s/sample", flush=True)
    print("GPU/PyTorch CPU/ONNX CPU numerical agreement: PASS", flush=True)
    print("Interface test only; this model is not a competition solution.", flush=True)


if __name__ == "__main__":
    main()
