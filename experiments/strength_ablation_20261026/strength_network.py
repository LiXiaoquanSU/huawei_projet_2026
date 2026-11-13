"""One ablation: bounded absolute-amplitude evidence at the hierarchical heads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared_hierarchy_20261025'))
from network import SharedHierarchy as OriginalHierarchy,crossroom_loss
import torch
from torch import nn
from torch.nn import functional as F

class SharedHierarchy(OriginalHierarchy):
    def __init__(self):
        super().__init__()
        # Preserve base initialization and subsequent random augmentation stream.
        rng=torch.get_rng_state()
        self.absolute=nn.Sequential(nn.Linear(6,32),nn.SiLU(),nn.Linear(32,5))
        nn.init.zeros_(self.absolute[-1].weight);nn.init.zeros_(self.absolute[-1].bias)
        torch.set_rng_state(rng)
    def tasks(self,t,s):
        task=super().tasks(t,s);z=task['logits'];a=s[:,:6].reshape(-1,2,3)
        mean=a.mean(-1);std=((a-mean[:,:,None]).square().mean(-1)+1e-6).sqrt()
        delta=.5*self.absolute(torch.cat([mean,std,a.amax(-1)],-1)).tanh()
        present=z[:,1:].logsumexp(-1,keepdim=True)-z[:,:1]+delta[:,:1]
        conditional=(z[:,1:].log_softmax(-1)+delta[:,1:]).log_softmax(-1)
        task['logits']=torch.cat([F.logsigmoid(-present),F.logsigmoid(present)+conditional],-1)
        return task
