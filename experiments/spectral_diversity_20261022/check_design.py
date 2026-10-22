import json
import torch
from train import load,build
from network import SpectralDiversity,ROOT
from model import signal_targets
from utils import task_loss
torch.set_num_threads(4);d=load();results={}
for v,base in (('diversity_context','context_events'),('diversity_cross','cross_events')):
    torch.manual_seed(42);m=build(v)
    torch.manual_seed(42);control=build(base)
    for k,w in control.state_dict().items():assert torch.equal(w,m.state_dict()[k]),k
    m.train();o=m.forward_tasks(d['t'][:2],d['s'][:2]);loss,_=task_loss(o,d['y'][:2],signal_targets(d['t'][:2]),'physical');loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():assert torch.isfinite(m.forward_features(torch.zeros_like(d['t'][:2]),torch.zeros_like(d['s'][:2]))).all()
    results[v]=dict(parameters=sum(p.numel() for p in m.parameters()),finite_gradient=True,base_initialization_identical=True,zero_input_finite=True)
x=torch.randn(2,24,150);q,_=torch.linalg.qr(torch.randn(24,24));error=float((SpectralDiversity.moments(x)-SpectralDiversity.moments(q@x)).abs().max());assert error<1e-5
zero=SpectralDiversity.moments(torch.zeros_like(x));assert torch.isfinite(zero).all() and zero.abs().max()==0
(ROOT/'checks/spectral_design_20261022.json').write_text(json.dumps(dict(variants=results,orthogonal_error=error,zero_moments=True),indent=2),encoding='utf-8')
print(results,error)
