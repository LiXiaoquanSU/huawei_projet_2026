"""Rx/time-symmetric covariance-moment crowd evidence, independent of CNN weights."""
import gc,json,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('adapter_shared',ROOT/'experiments/adapter049_20261024/run.py');shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
save,read,policy=shared.save,shared.read,shared.policy
WORK=Path('D:/huawei2026-experiments/correlation049_20261024')
OLD=Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')

class Descriptor(nn.Module):
    def stats(self,x):
        x=x-x.mean(-1,keepdim=True)
        energy=x.square().sum(-1,keepdim=True)
        normalized=x/(energy+1e-6).sqrt()
        c=normalized@normalized.transpose(-1,-2)
        # Fixed-shape diagonal mask avoids exporter control-flow subgraphs.
        diagonal=torch.eye(c.shape[-1],device=c.device,dtype=c.dtype)
        trace=(c*diagonal).sum((-1,-2)).clamp_min(1e-6)
        c=c/trace[...,None,None];c2=c@c
        tr2=c.square().sum((-1,-2));tr3=(c2*c).sum((-1,-2));tr4=c2.square().sum((-1,-2))
        terms=[torch.log1p(tr2*100),torch.log1p(tr3*1000),torch.log1p(tr4*10000)]
        for lag in (1,2,4,8,16,32):
            a,b=x[...,:-lag],x[...,lag:]
            ac=(a*b).sum((-1,-2))/(a.square().sum((-1,-2))*b.square().sum((-1,-2))+1e-10).sqrt()
            terms.append(ac)
        # Per-time energy concentration and symmetric half-window variation.
        e=x.square().mean(-2);e=e/e.mean(-1,keepdim=True).clamp_min(1e-8)
        terms.extend([e.square().mean(-1).log1p(),e.amax(-1).log1p(),(e[...,:75].mean(-1)-e[...,75:].mean(-1)).abs()])
        return torch.stack(terms,-1)
    def forward(self,t,s,absolute=True,include_clock=True):
        amp=t[:,384:768].reshape(-1,3,32,4,150).mean(3)
        phase=t[:,768:960].reshape(-1,3,64,150)
        features=[]
        for x in (amp,phase):
            a=self.stats(x);features.extend([a.mean(1),a.amax(1),a.amin(1),self.stats(x.flatten(1,2))])
            dx=x[...,1:]-x[...,:-1]
            dx=F.pad(dx,(0,1),mode='constant',value=0.)
            # Covariance moments of differences: omit the zero-padding-sensitive ACF.
            b=self.stats(dx)[...,:3];features.extend([b.mean(1),b.amax(1),b.amin(1)])
        if absolute:
            # Sort receiving channels within each statistic: no antenna identity.
            features.append(s[:,:12].reshape(-1,4,3).sort(-1).values.flatten(1))
        if include_clock:features.append(s[:,12:17])
        return torch.cat(features,-1)

class CountHead(nn.Module):
    def __init__(self,mean,std,ordinal=False):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std);self.ordinal=ordinal
        self.body=nn.Sequential(nn.Linear(len(mean),64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,4 if ordinal else 5))
    def forward(self,x):
        z=self.body(((x-self.mean)/self.std).clamp(-8,8))
        if not self.ordinal:return z.log_softmax(-1)
        p,_=shared.ordered_probabilities(z)
        return p.clamp_min(1e-8).log()

def metrics(z,y,base,ix):
    ix=torch.tensor(ix);p=z[ix].argmax(-1);b=base[ix].argmax(-1);yy=y[ix]
    cm=torch.bincount(yy*5+p,minlength=25).reshape(5,5)
    return dict(correct=int((p==yy).sum()),total=len(ix),corrected=int(((p==yy)&(b!=yy)).sum()),broken=int(((p!=yy)&(b==yy)).sum()),confusion_matrix=cm.tolist())

def train(x,y,fit_ix,cal_ix,ordinal,tag,seed=42):
    torch.manual_seed(seed);xx=x.cuda();yy=y.cuda();ix=torch.tensor(fit_ix,device='cuda')
    mean=xx[ix].mean(0);std=xx[ix].std(0).clamp_min(.05)
    model=CountHead(mean,std,ordinal).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    counts=torch.bincount(yy[ix],minlength=5).float();weights=1/counts[yy[ix]];weights/=weights.sum()
    snapshots={};history=[]
    for step in range(1,401):
        model.train();z=model(xx[ix]);loss=(F.nll_loss(z,yy[ix],reduction='none')*weights).sum()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step in (50,150,400):
            model.eval()
            with torch.no_grad():
                z=model(xx).cpu();ci=torch.tensor(cal_ix or fit_ix)
                cw=1/torch.bincount(y[ci],minlength=5).float()[y[ci]];cw/=cw.sum()
                nll=float((F.nll_loss(z[ci],y[ci],reduction='none')*cw).sum())
            snapshots[step]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};history.append(dict(step=step,calibration_nll=nll))
    chosen=min(history,key=lambda h:h['calibration_nll'])['step']
    model.load_state_dict(snapshots[chosen]);model.eval()
    with torch.no_grad():z=model(xx).cpu()
    torch.save(dict(state=snapshots[chosen],ordinal=ordinal,chosen_step=chosen,fit_indices=fit_ix,calibration_indices=cal_ix,history=history),WORK/f'{tag}.pt')
    return z,chosen,snapshots

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(architecture='Covariance normalized trace powers2/3/4, ACF lags1/2/4/8/16/32, energy concentration; per-Rx/pair symmetric reductions and joint correlations; small64/32 readout.',
      methods=['normalized','with_absolute'],heads=['softmax','conditional_ordinal'],steps=[50,150,400],seed=42,
      fitting='Teacher independent descriptor. Outer room excluded from head fitting and scaling. Select snapshot on existing whole-record calibration only. New branch has class-balanced CE, no transferred044bias.',
      selection='Explore fixed probability mixture.1/.25/.5 and keep01 guard separately. No final platform score inferred.',
      caveat='Rank is not literal human count. Dynamic covariance has no static-human supervision. Room/class confounding still present.'))
    d=torch.load(CACHE,weights_only=True);rows=d['rows'];y=d['y'];descriptor=Descriptor().cuda().eval();features={}
    with torch.no_grad():
        for absolute in (False,True):
            x=torch.cat([descriptor(d['t'][i:i+24].cuda(),d['s'][i:i+24].cuda(),absolute).cpu() for i in range(0,960,24)])
            assert torch.isfinite(x).all();features[absolute]=x;print('DESCRIPTOR',absolute,list(x.shape),flush=True)
    torch.save(dict(features=features,rows=rows,labels=y),WORK/'features.pt')
    reports=[];steps={}
    for room in ('a1','c1','g1','e1','actual'):
        actual=room=='actual'
        if actual:
            fit_ix=list(range(960));cal_ix=[];test=list(range(960));e=torch.load(Path('D:/huawei2026-experiments/presence_rescue_20261024/actual_evidence.pt'),weights_only=True);b=e['bias'];T=1.
        else:
            manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={k:{r['identifier'] for r in v} for k,v in manifest.items() if k in ('fit','calibration','test')}
            fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids['fit']];cal_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids['calibration']];test=[i for i,r in enumerate(rows) if r['identifier'] in ids['test']]
            e=torch.load(Path(f'D:/huawei2026-experiments/presence_rescue_20261024/proxy_evidence_{room}.pt'),weights_only=True)
            cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];b=torch.tensor(cal['bias']);T=cal['temperature']
        base=policy(e['logits'],b,T)
        for absolute in (False,True):
            for ordinal in (False,True):
                tag=f'{room}_abs{absolute}_ord{ordinal}';z,chosen,snapshots=train(features[absolute],y,fit_ix,cal_ix,ordinal,tag)
                if actual:
                    median=float(np.median(steps[absolute,ordinal]));chosen=min(snapshots,key=lambda s:abs(s-median));state=snapshots[chosen]
                    model=CountHead(state['mean'],state['std'],ordinal).eval();model.load_state_dict(state)
                    with torch.no_grad():z=model(features[absolute])
                    torch.save(dict(state=state,ordinal=ordinal,absolute=absolute,chosen_step=chosen,fit_indices=fit_ix),WORK/f'{tag}_final.pt')
                else:steps.setdefault((absolute,ordinal),[]).append(chosen)
                single=metrics(z,y,base,test);print('CORRELATION_HEAD',tag,chosen,single,flush=True)
                for mix in (.1,.25,.5):
                    candidate=((1-mix)*base.softmax(-1)+mix*z.exp()).clamp_min(1e-8).log()
                    for guard in ('none','keep01'):
                        zz=torch.where((base.argmax(-1)<=1)[:,None],base,candidate) if guard=='keep01' else candidate
                        result=dict(room=room,absolute=absolute,ordinal=ordinal,mix=mix,guard=guard,step=chosen,metrics=metrics(zz,y,base,test),single=single);reports.append(result)
                np.savez_compressed(WORK/f'{tag}.npz',logits=z.numpy(),base=base.numpy(),labels=y.numpy(),test_indices=test)
        save(WORK/'progress.json',reports)
    save(WORK/'results.json',reports);summary=[]
    for absolute in (False,True):
        for ordinal in (False,True):
            for mix in (.1,.25,.5):
                for guard in ('none','keep01'):
                    rr=[r for r in reports if r['absolute']==absolute and r['ordinal']==ordinal and r['mix']==mix and r['guard']==guard]
                    test=[r for r in rr if r['room']!='actual'];source=next(r for r in rr if r['room']=='actual');net=sum(r['metrics']['corrected']-r['metrics']['broken'] for r in test)
                    worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in test)
                    sourceok=all(source['metrics']['confusion_matrix'][k][k]>=old for k,old in enumerate([206,198,66,78,140]))
                    summary.append(dict(absolute=absolute,ordinal=ordinal,mix=mix,guard=guard,proxy_net=net,worst_room_delta=worst,room_net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in test},source=source['metrics'],eligible=net>0 and worst>=-.02 and sourceok))
    save(WORK/'summary.json',summary);print('ELIGIBLE',json.dumps([r for r in summary if r['eligible']]),flush=True)
if __name__=='__main__':main()
