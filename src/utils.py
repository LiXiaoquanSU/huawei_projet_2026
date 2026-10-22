import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def window(csi, ati, row):
    start, stop = int(row['start']), int(row['stop'])
    x = csi[start:stop].view(np.float32).reshape(600, 1, 3, 1024, 2).copy()
    dt = ati[start:stop].copy()
    dt[-1] = 0
    return x, dt

def task_loss(tasks,y,target,mode):
    ce=nn.functional.nll_loss(tasks['direct'].clamp_min(1e-8).log(),y)
    expert=nn.functional.cross_entropy(tasks['experts'].reshape(-1,5),y[:,None].expand(-1,4).reshape(-1))
    ordinal=js=aux=ce*0
    if mode!='direct':
        k=torch.arange(4,device=y.device)[None]
        mask=(y[:,None]>=k).to(tasks['logits'].dtype)
        labels=(y[:,None]>k).to(mask.dtype)
        binary=nn.functional.binary_cross_entropy_with_logits(tasks['conditional_logits'],labels,reduction='none')
        ordinal=((binary*mask).sum(0)/mask.sum(0).clamp_min(1)).mean()
        p,q=tasks['direct'].clamp_min(1e-7),tasks['ordinal'].clamp_min(1e-7);m=(p+q)*.5
        js=.5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1)).mean()
    if mode in ['auxiliary','physical']:aux=nn.functional.mse_loss(tasks['signal'],target)
    return ce+.15*expert+.5*ordinal+.1*js+.1*aux,dict(ce=ce.item(),expert=expert.item(),ordinal=ordinal.item(),consistency=js.item(),signal=aux.item())
