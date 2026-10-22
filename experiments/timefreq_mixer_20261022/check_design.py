import json
import torch
from engine import load
from network import ROOT,build,pack
torch.set_num_threads(4);d=load();t=d['t'][:2]
packed=pack(t,True)
restored=packed[:,:24].reshape(2,6,4,32,150).permute(0,1,3,2,4).reshape(2,768,150)
assert torch.equal(restored,t[:,:768]) and torch.equal(packed[:,24:],t[:,768:960].reshape(2,6,32,150))
results={}
for v in ('conv_coarse','conv_fine','axial_fine'):
    torch.manual_seed(42);m=build(v);assert not any(hasattr(m,k) for k in ('experts','ordinal','presence','static'))
    m.train();o=m.forward_features(t,d['s'][:2]);loss=torch.nn.functional.cross_entropy(o,d['y'][:2],label_smoothing=.05);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        for x,s in ((t,d['s'][:2]),(torch.zeros_like(t),torch.zeros_like(d['s'][:2]))):
            p=m.forward_features(x,s);assert p.shape==(2,5) and torch.isfinite(p).all();assert torch.allclose(p.exp().sum(1),torch.ones(2),atol=1e-5)
    results[v]=dict(parameters=sum(p.numel() for p in m.parameters()),finite_gradients=True,zero_input_finite=True)
(ROOT/'checks/timefreq_design_20261022.json').write_text(json.dumps(dict(lossless_repacking=True,variants=results),indent=2),encoding='utf-8');print(results)
