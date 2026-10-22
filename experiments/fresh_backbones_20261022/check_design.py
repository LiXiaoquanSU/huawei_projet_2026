import json
import torch
from engine import load
from network import build,ROOT,FreshModel
torch.set_num_threads(4);d=load();results={}
for v in ('band_patch','mixer','bigru'):
    torch.manual_seed(42);m=build(v);assert isinstance(m,FreshModel)
    assert not any(hasattr(m,k) for k in ('static','dynamic','phase','experts','ordinal','presence'))
    m.train();p=m.forward_features(d['t'][:2],d['s'][:2]);loss=torch.nn.functional.cross_entropy(p,d['y'][:2],label_smoothing=.05);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        for t,s in ((d['t'][:2],d['s'][:2]),(torch.zeros_like(d['t'][:2]),torch.zeros_like(d['s'][:2]))):
            out=m.forward_features(t,s);assert out.shape==(2,5) and torch.isfinite(out).all();assert torch.allclose(out.exp().sum(1),torch.ones(2),atol=1e-5)
    results[v]=dict(parameters=sum(p.numel() for p in m.parameters()),independent_backbone=True,finite_gradients=True,zero_input_finite=True)
(ROOT/'checks/fresh_design_20261022.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(results)
