"""Bounded zero-init residuals on nested out-of-record source predictions."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/detail_residual_20261027')
EVID=Path('D:/huawei2026-experiments/subnet_expectation_20261025/evidence')
OOF=Path('D:/huawei2026-experiments/protection_arbiter_20261025/crossfit')

class Residual(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,32),nn.SiLU(),nn.Linear(32,5));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        z=.35*self.net(x).tanh();return z-z.mean(-1,keepdim=True)

def metrics(z,base,y,ix):
    p=z[ix].argmax(-1);b=base[ix].argmax(-1);yy=y[ix];cm=torch.bincount(yy*5+p,minlength=25).reshape(5,5)
    return dict(correct=int(cm.trace()),total=len(ix),corrected=int(((p==yy)&(b!=yy)).sum()),broken=int(((p!=yy)&(b==yy)).sum()),confusion=cm.tolist())

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    data=Path('D:/huawei2026-experiments/detail_features_20261027');rows=old.read(data/'rows.json');y=torch.tensor([r['headcount'] for r in rows]);d=np.load(data/'features.npz');s=np.load('D:/huawei2026-experiments/next053_20261025/s.npy')[:,0];ss=s[:,:12].reshape(-1,4,3);summary=np.concatenate([ss.mean(-1),ss.std(-1),s[:,12:]],1)
    xs={'summary':summary,'time':np.concatenate([summary,d['temporal']],1),'phase':np.concatenate([summary,d['phase']],1),'both':np.concatenate([summary,d['temporal'],d['phase']],1)}
    xs['capacity_control']=np.concatenate([summary,np.zeros((len(summary),320),dtype=np.float32)],1)
    old.save(WORK/'protocol.json',dict(folds=['a1','c1','g1','e1'],seeds=[42,2026],variants=list(xs),epochs=60,zero_init=True,pairwise_logit_change_bound=.7,selection='source calibrated NLL; checkpoints every5 epochs; lambda 0,.25,.5,1; source calibration corrected>=broken; zero-residual baseline always eligible',loss='class-room weighted NLL + .2 KL on initially correct source OOF samples',source_training='three cached inner teachers excluded each residual-training record; each also excluded outer room',limitation='Teacher bias used source calibration labels; same calibration reused for residual selection, not an independent validation estimate. Only outer rooms are excluded throughout.',capacity_control='Added after initial32 fits: same333-input architecture/initialization as both, but320 detail inputs fixedzero; no new information. This matches parameter count, not active input dimension.'))
    result=old.read(WORK/'results.json') if (WORK/'results.json').exists() else {}
    for room in ('a1','c1','g1','e1'):
        ref=torch.load(EVID/f'{room}.pt',weights_only=True);assert ref['rows']==rows;b,T=old.calibration(room);base=old.policy(ref['raw'],b,T).detach();fit,cal,test=old.split(room,rows);trainbase=base.clone();seen=[]
        for k in range(3):
            v=torch.load(OOF/room/f'fold{k}/predictions.pt',weights_only=True);ix=v['indices'];seen+=ix
            trainids=set(v['fit_records']);heldids=set(v['oof_records']);calids=set(v['calibration_records']);assert not trainids&heldids and not calids&heldids
            assert all(rows[i]['room']!=room for i in ix);assert not ({rows[i]['identifier'] for i in test}&(trainids|heldids|calids))
            trainbase[ix]=old.policy(v['raw'],v['bias'],v['temperature'])
        assert sorted(seen)==sorted(fit) and len(seen)==len(set(seen))
        w=old.balance(rows,y,fit);cw=1/torch.bincount(y[cal],minlength=5)[y[cal]].float();cw/=cw.sum();baseline_nll=float((F.cross_entropy(base[cal],y[cal],reduction='none')*cw).sum());basecal=metrics(base,base,y,cal)
        for variant,features in xs.items():
          scaler=StandardScaler().fit(features[fit]);x=torch.tensor(scaler.transform(features).clip(-8,8),dtype=torch.float32)
          for seed in (42,2026):
            name=f'{room}_{variant}_s{seed}'
            if name in result:continue
            torch.manual_seed(seed);m=Residual(x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);initial=copy.deepcopy(m.state_dict());best=dict(nll=baseline_nll,epoch=0,alpha=0.,state=initial);history=[]
            with torch.no_grad():assert torch.equal(m(x),torch.zeros(len(x),5))
            for epoch in range(1,61):
                m.train();delta=m(x[fit]);z=trainbase[fit]+delta;pp=trainbase[fit].softmax(-1);keep=(trainbase[fit].argmax(-1)==y[fit]).float()
                kl=(pp*(pp.clamp_min(1e-12).log()-z.log_softmax(-1))).sum(-1)
                loss=(F.cross_entropy(z,y[fit],reduction='none')*w).sum()+.2*(kl*keep*w).sum()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if epoch%5:continue
                m.eval()
                with torch.no_grad():delta=m(x);assert float((delta.amax(1)-delta.amin(1)).max())<=.70001
                for alpha in (.25,.5,1.):
                    z=base+alpha*delta;nll=float((F.cross_entropy(z[cal],y[cal],reduction='none')*cw).sum());cm=metrics(z,base,y,cal);history.append(dict(epoch=epoch,alpha=alpha,nll=nll,calibration=cm))
                    if cm['correct']>=basecal['correct'] and nll<best['nll']:best=dict(nll=nll,epoch=epoch,alpha=alpha,state=copy.deepcopy(m.state_dict()))
            m.load_state_dict(best['state']);m.eval()
            with torch.no_grad():z=base+best['alpha']*m(x)
            name=f'{room}_{variant}_s{seed}';r=dict(room=room,variant=variant,seed=seed,epoch=best['epoch'],alpha=best['alpha'],nll=best['nll'],baseline_nll=baseline_nll,baseline=metrics(base,base,y,test),test=metrics(z,base,y,test),history=history)
            torch.save(dict(state=best['state'],scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),alpha=best['alpha'],logits=z,baseline=base,labels=y,test_indices=test,fit_indices=fit,calibration_indices=cal),WORK/f'{name}.pt');result[name]=r;old.save(WORK/'results.json',result);print(name,best['epoch'],best['alpha'],r['test'],flush=True)
    agg={}
    for variant in xs:
      for seed in (42,2026):
        rr=[v for v in result.values() if v['variant']==variant and v['seed']==seed];agg[f'{variant}_s{seed}']={k:sum(v['test'][k] for v in rr) for k in ('correct','total','corrected','broken')}
    old.save(WORK/'aggregate.json',agg);print('TOTAL',json.dumps(agg),flush=True)
if __name__=='__main__':main()
