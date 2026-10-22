"""Fixed low-weight public-data experts: exploratory follow-up, no weight search."""
import csv,json
import numpy as np
import torch
from torch import nn
from net import ROOT,TemporalBank,bank_input
from prepare import WORK,save
from run import CACHE,read,FOLDS
def main():
    torch.set_num_threads(4);d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
    experts=[]
    for s in (42,2026):
        state=torch.load(WORK/'pretrain'/f'external_multitask_s{s}_public/final.pt',map_location='cpu',weights_only=True)
        bank=TemporalBank().eval();bank.load_state_dict(state['bank']);head=nn.Linear(256,5).eval();head.load_state_dict(state['count']);experts.append((bank,head))
    cm=np.zeros((5,5),dtype=np.int64);basecm=cm.copy();folds={};bf={};nll=[]
    folder=WORK/'anchor_expert';folder.mkdir(exist_ok=True)
    for fold in FOLDS:
        path=__import__('pathlib').Path('D:/huawei2026-experiments/motion_external_20261021/fits')/f'baseline_s42_{fold}/predictions.csv'
        with path.open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        vi=[i for i,r in enumerate(d['rows']) if r['room']==fold]
        assert [(r['identifier'],int(r['start'])) for r in rr]==[(d['rows'][i]['identifier'],int(d['rows'][i]['start'])) for i in vi]
        probabilities=[]
        with torch.no_grad():
            for start in range(0,len(vi),32):
                x=bank_input(d['t'][vi[start:start+32]])
                probabilities.append(sum(head(bank(x)).softmax(1) for bank,head in experts).numpy()/2)
        ep=np.concatenate(probabilities);base=np.exp(np.asarray([[float(r[f'log_probability_{k}']) for k in range(5)] for r in rr]));p=.9*base+.1*ep;y=d['y'][vi].numpy()
        np.add.at(cm,(y,p.argmax(1)),1);np.add.at(basecm,(y,base.argmax(1)),1);folds[fold]=float(np.mean(y==p.argmax(1)));bf[fold]=float(np.mean(y==base.argmax(1)));nll.append(float(-np.log(p[np.arange(len(y)),y].clip(1e-8)).mean()))
        np.savez_compressed(folder/f'{fold}_predictions.npz',probabilities=p,external_probabilities=ep,base_probabilities=base,y=y)
    mean=float(np.mean(list(folds.values())));macro=float(np.mean(np.diag(cm)/cm.sum(1)));score=min(mean,macro);bscore=min(float(np.mean(list(bf.values()))),float(np.mean(np.diag(basecm)/basecm.sum(1))))
    result=dict(variant='anchor_external_experts',external_seeds=[42,2026],base_weight=.9,external_weight=.1,room_mean=mean,macro_recall=macro,selection_score=score,worst_room=min(folds.values()),rooms=folds,nll=float(np.mean(nll)),confusion_matrix=cm.tolist(),anchor_selection_score=bscore,anchor_rooms=bf,selected=bool(score>=bscore+.005 and min(folds.values())>=min(bf.values())),note='Exploratory fixed-weight follow-up after the first training screen. Experts have never trained on official labels.')
    save(WORK/'anchor_expert_decision.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
