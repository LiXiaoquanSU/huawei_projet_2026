"""Predeclared end-to-end architecture experiment, no old learned weights or U labels."""
import sys,json,copy,random,gc,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from network import EvidenceChain
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/static_texture_20261029')
CACHE=Path('D:/huawei2026-experiments/next053_20261025')
ROOMS=('a1','c1','g1','e1');FOLDS=(*ROOMS,'record0','record1','record2');EPOCHS=20

def predict(m,t,s,indices):
    m.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(indices),24):
            jj=indices[start:start+24];z=m(t[jj].flatten(0,1),s[jj].flatten(0,1)).reshape(-1,2,5)
            out.append(z.exp().mean(1).clamp_min(1e-12).log().cpu())
    return torch.cat(out)

def metric(z,y):
    cm=torch.bincount(y*5+z.argmax(-1),minlength=25).reshape(5,5)
    return dict(correct=int(cm.trace()),total=len(y),accuracy=float(cm.trace()/len(y)),confusion=cm.tolist())

def invariance(m,t,s):
    m.eval();error=0.
    with torch.inference_mode():
        ref=m(t,s).exp()
        for perm in old.PERMS:
            ix,sg,si=[v.to(t.device) for v in old.permutation_indices(perm)]
            error=max(error,float((m(t.index_select(1,ix)*sg,torch.cat([s.index_select(1,si),s[:,17:]],-1)).exp()-ref).abs().max()))
    assert error<1e-5,error
    return error

def split(rows,fold):
    if fold in ROOMS:return old.split(fold,rows)
    r=old.read(Path('D:/huawei2026-experiments/detail_features_20261027/probe_results.json'))['folds'][fold]
    return tuple([i for i,v in enumerate(rows) if v['identifier'] in set(r[k])] for k in ('fit_records','calibration_records','test_records'))

def sample_weights(rows,y,fit):
    # Each count gets equal mass, then room, then recording, then window.
    keys=[(int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room'],rows[i]['identifier']) for i in fit]
    groups={k:keys.count(k) for k in set(keys)};rooms={c:len({r for cc,r,_ in groups if cc==c}) for c in range(5)}
    counts={(c,r):len({i for cc,rr,i in groups if cc==c and rr==r}) for c,r,_ in groups}
    w=torch.tensor([1/(groups[k]*rooms[k[0]]*counts[k[:2]]) for k in keys]);return w/w.sum()

def fit(fold,variant,seed,rows,t,s,y,full=False):
    folder=WORK/f'{fold}_{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    fi,cal,test=(list(range(len(rows))),[],list(range(len(rows)))) if full else split(rows,fold)
    ids=lambda ix:sorted({rows[i]['identifier'] for i in ix})
    if not full:assert not(set(ids(fi))&set(ids(cal)) or set(ids(fi))&set(ids(test)) or set(ids(cal))&set(ids(test)))
    if fold in ROOMS:assert all(rows[i]['room']!=fold for i in fi+cal)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    m=EvidenceChain(variant).cuda();inv=invariance(m,t[:2,0],s[:2,0]);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.02)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00003);w=sample_weights(rows,y.cpu(),fi);gen=torch.Generator().manual_seed(seed)
    chosen_epoch=old.read(WORK/'selection.json')['epochs'] if full else None;best=None;history=[]
    for epoch in range(1,EPOCHS+1):
        m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[]
        for jj in draw.split(24):
            z=m(t[jj].flatten(0,1),s[jj].flatten(0,1));yy=y[jj,None].expand(-1,2).flatten();p=z.exp().reshape(-1,2,5);q=p.mean(1,keepdim=True)
            loss=F.nll_loss(z,yy)+.05*(p*(p.clamp_min(1e-8).log()-q.clamp_min(1e-8).log())).sum(-1).mean()
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
        sched.step();e=dict(epoch=epoch,loss=float(np.mean(losses)))
        if cal:
            z=predict(m,t,s,cal);yy=y[cal].cpu();cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum();nll=float((-z[range(len(cal)),yy]*cw).sum());e.update(calibration_nll=nll,calibration=metric(z,yy))
            if best is None or nll<best['nll']:best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}))
        elif epoch==chosen_epoch:best=dict(epoch=epoch,nll=None,state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}))
        if epoch==1 or epoch%5==0:
            z=predict(m,t,s,fi);e['fit']=metric(z,y[fi].cpu());print('TRAIN',folder.name,epoch,'loss',e['loss'],'fit',e['fit']['correct'],'cal',e.get('calibration',{}).get('correct'),flush=True)
        history.append(e)
    last=predict(m,t,s,list(range(len(rows))));torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=EPOCHS,variant=variant,seed=seed,fit_records=ids(fi),calibration_records=ids(cal),test_records=ids(test)),folder/'last.pt')
    m.load_state_dict(best['state']);err=invariance(m,t[:2,0],s[:2,0]);z=predict(m,t,s,list(range(len(rows))))
    with torch.inference_mode():
        tasks=m.tasks(t[:24,0],s[:24,0]);mechanism=dict(mean_absolute_modulation=float(tasks['modulation'].abs().mean()),mean_local_probability_std=float(tasks['local_probabilities'].std(-1).mean()))
    torch.save(dict(model_state=best['state'],epoch=best['epoch'],variant=variant,seed=seed,fit_records=ids(fi),calibration_records=ids(cal),test_records=ids(test)),folder/'final.pt')
    torch.save(dict(logits=z,last_logits=last,labels=y.cpu(),rows=rows,test_indices=test,calibration_indices=cal,fit_indices=fi),folder/'predictions.pt')
    r=dict(fold=fold,variant=variant,seed=seed,epoch=best['epoch'],calibration_nll=best['nll'],parameters=sum(p.numel() for p in m.parameters()),test=metric(z[test],y[test].cpu()),last_test=metric(last[test],y[test].cpu()),fit=metric(z[fi],y[fi].cpu()),history=history,mechanism=mechanism,initial_invariance=inv,final_invariance=err,sha256=old.sha(folder/'final.pt'))
    old.save(folder/'result.json',r);print('DONE',folder.name,'epoch',best['epoch'],r['test'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    WORK.mkdir(parents=True,exist_ok=True);rows=old.read(CACHE/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows],device='cuda')
    t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda()
    curvature=torch.from_numpy(np.load('D:/huawei2026-experiments/static_phase_20261029/features.npz')['static_phase']).cuda();s=torch.cat([s,curvature[:,None,:].expand(-1,2,-1)],-1)
    if '--full' in sys.argv:
        for seed in (42,):fit('actual','full',seed,rows,t,s,y,True)
        return
    old.save(WORK/'protocol.json',dict(architecture='Texture network plus static relative phase curvature, with matched-capacity zero-input control',epochs=EPOCHS,lr=.0003,weight_decay=.02,batch=24,variants=['full','zero'],seeds={'full':[42],'zero':[42]},folds=list(FOLDS),selection='per-fit minimum source calibration balanced NLL; primary full architecture and delivery seed42 fixed in advance; final epochs median source-selected epochs across7 full fits, rounded down; all full fits run20 and preserve last checkpoint',data='48 labeled records only; exact original preprocessing cached; no old learned weights, U, external data or pseudo labels',loss='NLL + .05 reversalJS',sampling='class then room then record then window balanced',limitations='Reused development folds; fixed existing preprocessing means raw static phase curvature added; no raw validity quality head; same old dynamic preprocessing',hashes={k:old.sha(CACHE/k) for k in ('rows.json','t.npy','s.npy')}))
    for variant in ('full','zero'):
        for fold in (FOLDS[4:] if '--record' in sys.argv else FOLDS):
            for seed in (42,):fit(fold,variant,seed,rows,t,s,y)
    if '--record' in sys.argv:return
    values=[old.read(WORK/f'{f}_full_s{seed}'/'result.json')['epoch'] for f in FOLDS for seed in (42,)]
    old.save(WORK/'selection.json',dict(variant='full',delivery_seed=42,seeds=[42],epochs=int(np.median(values)),source_epochs=values,reason='Primary hypothesis and seed fixed before evaluation; epochs selected only on source calibration; no choice by target accuracy',platform_score=None))
if __name__=='__main__':main()
