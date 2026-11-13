"""Two source calibration partitions, two seeds, four held-out rooms."""
import sys,json,copy,hashlib,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from network import CovarianceFeatures,CovarianceNet
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/covariance_20261026')
CACHE=Path('D:/huawei2026-experiments/next053_20261025')
ROOMS=('a1','c1','g1','e1')

def split(rows,room,partition):
    if partition==0:return old.split(room,rows)
    eligible=[i for i,r in enumerate(rows) if r['room'] not in (room,'unknown')]
    ids={r['identifier']:int(r['headcount']) for r in rows if r['room'] not in (room,'unknown')}
    chosen=set()
    for label in range(5):
        options=sorted([i for i,c in ids.items() if c==label],key=lambda s:hashlib.sha256(('calibration17:'+s).encode()).hexdigest())
        assert len(options)>=2;chosen.add(options[1])
    return [i for i in eligible if rows[i]['identifier'] not in chosen],[i for i in eligible if rows[i]['identifier'] in chosen],[i for i,r in enumerate(rows) if r['room']==room]

def predict(m,x):
    m.eval()
    with torch.inference_mode():return torch.cat([m(xx.flatten(0,1)).exp().reshape(-1,2,5).mean(1).clamp_min(1e-12).log().cpu() for xx in x.split(64)])

def metrics(z,y,ix):
    cm=torch.bincount(y[ix]*5+z[ix].argmax(-1),minlength=25).reshape(5,5)
    return dict(correct=int(cm.trace()),total=len(ix),confusion=cm.tolist())

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    WORK.mkdir(parents=True,exist_ok=True);rows=old.read(CACHE/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows]);features=CovarianceFeatures().cuda()
    if not (WORK/'kernels.pt').exists():
        t=np.load(CACHE/'t.npy',mmap_mode='r');out=[]
        with torch.inference_mode():
            for start in range(0,len(t),16):
                v=torch.tensor(np.array(t[start:start+16]),device='cuda');out.append(features(v.flatten(0,1)).reshape(-1,2,8,30,30).cpu())
        x=torch.cat(out);assert torch.isfinite(x).all();torch.save(x,WORK/'kernels.pt')
        v=torch.tensor(np.array(t[:2,0]),device='cuda');err=0.
        with torch.inference_mode():
            for perm in old.PERMS:
                ix,sg,si=[a.cuda() for a in old.permutation_indices(perm)]
                err=max(err,float((features(v)-features(v.index_select(1,ix)*sg)).abs().max()))
        assert err<1e-4;old.save(WORK/'feature_check.json',dict(receiver_permutation_max_error=err,shape=list(x.shape)))
    x=torch.load(WORK/'kernels.pt',weights_only=True).cuda()
    old.save(WORK/'protocol.json',dict(seeds=[42,2026],calibration_partitions=2,epochs=40,minimum_selection_epoch=5,selection='source class-balanced NLL only; two distinct whole-record calibration sets; target labels only used for final development evaluation',lr=.001,weight_decay=.02,loss='NLL + .05 forward/reverse JS',absolute_amplitude=False,external_data=False,platform_score=None))
    for room in ROOMS:
      for partition in (0,1):
       for seed in (42,2026):
        folder=WORK/f'{room}_p{partition}_s{seed}';folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():continue
        fit,cal,test=split(rows,room,partition); ids=lambda ix:sorted({rows[i]['identifier'] for i in ix})
        assert not set(ids(fit))&set(ids(cal));assert not set(ids(fit)+ids(cal))&set(ids(test))
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        m=CovarianceNet().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.0001)
        w=old.balance(rows,y,fit);gen=torch.Generator().manual_seed(seed);best=None;unrestricted=None;history=[]
        for epoch in range(1,41):
            m.train();draw=torch.tensor(fit)[torch.multinomial(w,len(fit),replacement=True,generator=gen)];losses=[]
            for jj in draw.split(32):
                z=m(x[jj].flatten(0,1));yy=y[jj,None].expand(-1,2).flatten().cuda();p=z.exp().reshape(-1,2,5);avg=p.mean(1,keepdim=True)
                loss=F.nll_loss(z,yy)+.05*(p*(p.clamp_min(1e-8).log()-avg.clamp_min(1e-8).log())).sum(-1).mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
            scheduler.step();z=predict(m,x);cw=1/torch.bincount(y[cal],minlength=5)[y[cal]].float();cw/=cw.sum();nll=float((-z[cal,y[cal]]*cw).sum())
            entry=dict(epoch=epoch,loss=float(np.mean(losses)),calibration_nll=nll,fit=metrics(z,y,fit),calibration=metrics(z,y,cal));history.append(entry)
            if unrestricted is None or nll<unrestricted['nll']:unrestricted=dict(epoch=epoch,nll=nll)
            if epoch>=5 and (best is None or nll<best['nll']):best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.cpu() for k,v in m.state_dict().items()}))
        m.load_state_dict(best['state']);z=predict(m,x)
        torch.save(dict(model_state=best['state'],epoch=best['epoch'],seed=seed,fit_records=ids(fit),calibration_records=ids(cal),test_records=ids(test)),folder/'final.pt')
        torch.save(dict(logits=z,labels=y,test_indices=test,rows=rows),folder/'predictions.pt')
        result=dict(room=room,partition=partition,seed=seed,epoch=best['epoch'],unrestricted_selection=unrestricted,history=history,test=metrics(z,y,test),checkpoint_sha256=old.sha(folder/'final.pt'))
        old.save(folder/'result.json',result);print('DONE',folder.name,'epoch',best['epoch'],result['test'],flush=True)
    totals={}
    for seed in (42,2026):
      for part in (0,1,'mean'):
        records=[];cm=torch.zeros(5,5,dtype=torch.long)
        for room in ROOMS:
            ds=[torch.load(WORK/f'{room}_p{p}_s{seed}'/'predictions.pt',weights_only=True) for p in (0,1)]
            z=torch.stack([d['logits'].exp() for d in ds]).mean(0).log() if part=='mean' else ds[part]['logits'];ix=ds[0]['test_indices'];m=metrics(z,y,ix);cm+=torch.tensor(m['confusion']);records.append(dict(room=room,**m))
        totals[f's{seed}_p{part}']=dict(correct=int(cm.trace()),total=880,confusion=cm.tolist(),rooms=records)
    old.save(WORK/'results.json',totals);print('TOTAL',json.dumps(totals),flush=True)

if __name__=='__main__':main()
