"""Equal-weight fusion of complementary trained development models."""
from pathlib import Path
import torch
from torch import nn
from improved_models import ImprovedModel
from tf_model import TFModel

MEMBERS=['whiten','tfcnn','tfcnn_mixstyle']


def member_path(name,room,seed=42):
    root='iterations_v2' if name=='whiten' else 'iterations_tf'
    return Path(f'artifacts/{root}/{name}_{room}_seed{seed}/best.pt')


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.models=nn.ModuleList([ImprovedModel('whiten'),TFModel('tfcnn'),TFModel('tfcnn_mixstyle')])

    def load_room(self,room):
        for name,model in zip(MEMBERS,self.models):
            state=torch.load(member_path(name,room),map_location='cpu',weights_only=True)
            model.load_state_dict(state['model_state'])
        return self.eval()

    def forward(self,frames,intervals):
        scores=[]
        for model in self.models:
            logits=model(frames,intervals)
            center=logits-logits.mean(-1,keepdim=True)
            normalized=center/torch.sqrt(center.square().mean(-1,keepdim=True)).clamp_min(.01)
            scores.append(torch.softmax(normalized,dim=-1))
        # Log probabilities are valid logits and preserve the fusion's argmax.
        return torch.log(torch.stack(scores).mean(0).clamp_min(1e-8))
