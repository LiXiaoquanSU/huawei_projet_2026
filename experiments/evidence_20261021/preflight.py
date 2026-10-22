"""Check signal-axis identity and actual-time attention invariance."""
import torch
from network import PairEvidence,TimedAttention,build
torch.set_num_threads(4)
pairs=PairEvidence().eval();captured=[]
hook=pairs.encode.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
a=torch.zeros(1,3,128,150);p=torch.zeros(1,6,32,150);t=torch.zeros(1,3,150)
for rx in range(3):
    for segment in range(5):a[:,rx,:,segment*30:(segment+1)*30]=10*rx+segment
for ch in range(6):p[:,ch]=100+ch
with torch.no_grad():z=pairs.tokens(a,p,t)
assert z.shape==(1,240,256)
x=captured[0]
assert x.shape==(1,3,16,5,18)
for k,(u,v) in enumerate(((0,1),(0,2),(1,2))):
    for seg in range(5):
        expected=torch.tensor([10*u+seg,10*v+seg,100+2*k,101+2*k,10*(u-v),(10*u+seg)*(10*v+seg)],dtype=x.dtype)
        assert torch.allclose(x[0,k,:,seg,:6],expected.expand(16,6))
hook.remove()
attention=TimedAttention().eval();x=torch.randn(2,11,256);clock=torch.arange(11)[None].expand(2,-1)*.02
with torch.no_grad():
    assert torch.allclose(attention(x,clock),attention(x,clock+10),atol=2e-6,rtol=1e-5)
    assert not torch.allclose(attention(x,clock),attention(x,clock*20),atol=1e-5)
print('PASS: pair direction, frequency/segment identity, elapsed-time offset invariance and gap sensitivity')
