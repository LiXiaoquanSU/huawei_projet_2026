"""Fixed0.9 anchor+0.1 single-Rx2D external expert; exploratory only."""
import csv,json
import numpy as np
import torch
from torch import nn
from train import ROOT,WORK,CACHE,TRIAD,EvidenceEncoder,read
from acquire import save
class Expert(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=EvidenceEncoder(1,[64,128,256,384],256,style=False,frequency_bins=None);self.head=nn.Linear(256,5)
    def forward(self,t):
        a=t[:,384:768].reshape(-1,1,128,150);a=a-a.mean(-1,keepdim=True);a=(a/(a.square().mean((1,2,3),keepdim=True)+1e-6).sqrt()).clamp(-6,6)
        return self.head(self.encoder(a)).softmax(-1).reshape(-1,3,5).mean(1)
def loaded():
    m=Expert().eval();s=torch.load(WORK/'pretrain/ehu_transfer_s42_public/final.pt',map_location='cpu',weights_only=True);m.encoder.load_state_dict(s['encoder']);m.head.load_state_dict(s['head']);return m
def main():
    torch.set_num_threads(4);model=loaded();d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None);cm=np.zeros((5,5),np.int64);basecm=cm.copy();folds={};bf={};nll=[]
    for fold in TRIAD:
        path=__import__('pathlib').Path('D:/huawei2026-experiments/motion_external_20261021/fits')/f'baseline_s42_{fold}/predictions.csv'
        with path.open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        vi=[i for i,r in enumerate(d['rows']) if r['room']==fold];assert [(r['identifier'],int(r['start'])) for r in rr]==[(d['rows'][i]['identifier'],int(d['rows'][i]['start'])) for i in vi]
        pieces=[]
        with torch.no_grad():
            for start in range(0,len(vi),16):pieces.append(model(d['t'][vi[start:start+16]]).numpy())
        ep=np.concatenate(pieces);bp=np.exp(np.asarray([[float(r[f'log_probability_{k}']) for k in range(5)] for r in rr]));p=.9*bp+.1*ep;y=d['y'][vi].numpy();np.add.at(cm,(y,p.argmax(1)),1);np.add.at(basecm,(y,bp.argmax(1)),1)
        folds[fold]=float(np.mean(y==p.argmax(1)));bf[fold]=float(np.mean(y==bp.argmax(1)));nll.append(float(-np.log(p[np.arange(len(y)),y].clip(1e-8)).mean()))
    mean=float(np.mean(list(folds.values())));macro=float(np.mean(np.diag(cm)/cm.sum(1)));bm=float(np.mean(np.diag(basecm)/basecm.sum(1)));base=min(float(np.mean(list(bf.values()))),bm);score=min(mean,macro)
    result=dict(variant='anchor_ehu_expert',room_mean=mean,macro_recall=macro,selection_score=score,worst=min(folds.values()),rooms=folds,nll=float(np.mean(nll)),confusion_matrix=cm.tolist(),base_selection_score=base,base_rooms=bf,selected=bool(score>=base+.005 and macro>=bm and min(folds.values())>=min(bf.values())),base_weight=.9,expert_weight=.1,expert_parameters=sum(p.numel() for p in model.parameters()),note='One fixed-weight exploratory test; expert never trained on official labels; validation anchor models trained only on their source folds.')
    save(WORK/'expert_decision.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
