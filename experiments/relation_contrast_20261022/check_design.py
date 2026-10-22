"""Check relation construction and finite gradients on actual cached data."""
import json
import torch
from train import load
from network import ROOT,RelationDynamics,build

torch.set_num_threads(4);torch.manual_seed(42)
x=torch.randn(2,72,150)
c,e=RelationDynamics.correlation(x)
shifted,_=RelationDynamics.correlation(x+torch.randn(2,72,1)*3)
assert (c-shifted).abs().max()<1e-5
assert (c-c.transpose(-1,-2)).abs().max()<1e-6
zero,_=RelationDynamics.correlation(torch.zeros_like(x));assert torch.isfinite(zero).all() and zero.abs().max()==0
d=load();results={}
for v in ('relation_context','contrast_context','relation_only'):
    torch.manual_seed(42);m=build(v).train();z=m.forward_features(d['t'][:2],d['s'][:2]);assert torch.isfinite(z).all()
    (-z[:,2].mean()).backward()
    grads=[p.grad for n,p in m.named_parameters() if 'temporal' in n and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum()>0 for g in grads)
    m.eval()
    with torch.no_grad():
        for t,s in ((d['t'][:2],d['s'][:2]),(torch.zeros_like(d['t'][:2]),torch.zeros_like(d['s'][:2]))):
            assert torch.isfinite(m.forward_features(t,s)).all()
    results[v]={'finite_gradients':True,'zero_input_finite':True,'parameters':sum(p.numel() for p in m.parameters())}
(ROOT/'checks/relation_design_20261022.json').write_text(json.dumps(dict(correlation_offset_invariance=True,variants=results),indent=2),encoding='utf-8')
print(results)
