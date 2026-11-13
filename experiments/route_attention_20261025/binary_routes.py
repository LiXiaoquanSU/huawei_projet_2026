"""Amendment after failed unrestricted residuals: relative-view binary correction."""
from ra_common import *

def relative_features(e,raw,base):
    # Pairwise cosine relations remove the absolute magnitude and feature mean.
    u=F.normalize(e,dim=-1);u=u.transpose(1,2);g=u@u.transpose(-1,-2)
    mask=~torch.eye(12,dtype=torch.bool,device=e.device);off=g[:,:,mask]
    parts=[off.mean(-1),off.std(-1,unbiased=False),off.amin(-1),off.amax(-1)]
    paired=(u[:,:,::2]*u[:,:,1::2]).sum(-1)
    parts += [paired.mean(-1),paired.std(-1,unbiased=False),paired.amin(-1),paired.amax(-1)]
    parts += [torch.stack([(off[:,a]-off[:,b]).square().mean(-1) for a,b in ((0,1),(0,2),(1,2))],-1)]
    # Calibrated base-to-view relative odds, never an explicit room identity.
    odds=(raw[...,1]-raw[...,0]).flatten(1);bo=base[:,1]-base[:,0];rel=(odds-odds.mean(-1,keepdim=True)).clamp(-8,8)
    pair=(odds[:,::2]-odds[:,1::2]).abs()
    parts += [torch.stack([rel.std(-1,unbiased=False),rel.amin(-1),rel.amax(-1),pair.mean(-1),pair.std(-1,unbiased=False),bo.clamp(-8,8),bo.tanh()],-1)]
    return torch.cat(parts,-1)

class RelativeBinary(nn.Module):
    def __init__(self,architecture='linear'):
        super().__init__();self.architecture=architecture
        self.net=nn.Linear(34,1,bias=False) if architecture=='linear' else nn.Sequential(nn.Linear(34,16),nn.SiLU(),nn.Dropout(.2),nn.Linear(16,1,bias=False))
        last=self.net if architecture=='linear' else self.net[-1];nn.init.zeros_(last.weight)
        self.register_buffer('center',torch.zeros(34));self.register_buffer('scale',torch.ones(34))
    def forward(self,x):return self.net(((x-self.center)/self.scale).clamp(-4,4)).squeeze(-1).tanh()

def binary_deploy(base,shift,amount):
    delta=torch.stack([-shift,shift,shift*0,shift*0,shift*0],-1)*amount
    proposed=base+delta
    # This specialist cannot promote a 0/1 prediction into a higher count.
    return torch.where(((base.argmax(-1)<2)&(proposed.argmax(-1)<2))[:,None],proposed,base)

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    import time
    save(WORK/'binary_plan.json',dict(reason='All40 route residual fits failed held-room criterion, often promoted g1 lowcount tohighcount. New specialist onlycorrects0vs1, usesrelativeview relations, highcount promotion veto.',architectures=['linear','mlp'],objectives=['balanced','dro'],seeds=[42,2026],steps=[50,150,400],amounts=[.25,.5,1.],selection='same criteria as initialplan; no targetroom labels in fit; actualstep derivedfrom proxycalibration; development is reused, not blind'))
    for room in ('a1','c1','g1','e1','actual'):
      d=torch.load(WORK/f'{room}.pt',weights_only=True);prior=torch.load(PREV/f'{room}.pt',weights_only=True);x=relative_features(d['e'],prior['raw'],d['base']);torch.save(x,WORK/f'binary_x_{room}.pt')
      fit,cal,test=old.split(room,d['rows']);fit=[i for i in fit if int(d['labels'][i])<2];cal=[i for i in cal if int(d['labels'][i])<2]
      xc=x.cuda();base=d['base'].cuda();y=d['labels'].cuda();fi=torch.tensor(fit,device='cuda');ci=torch.tensor(cal,device='cuda');groups=sorted({(int(y[i]),d['rows'][i]['room']) for i in fit});gi=torch.tensor([groups.index((int(y[i]),d['rows'][i]['room'])) for i in fit],device='cuda');ng=torch.bincount(gi,minlength=len(groups));priorw=torch.tensor([1/sum(c==c2 for c2,r2 in groups) for c,r in groups],device='cuda');priorw/=priorw.sum()
      for arch in ('linear','mlp'):
       for obj in ('balanced','dro'):
        for seed in (42,2026):
          dest=WORK/'binary_fits'/f'{arch}_{obj}_{seed}_{room}';dest.mkdir(parents=True,exist_ok=True)
          if (dest/'complete.json').exists():continue
          torch.manual_seed(seed);m=RelativeBinary(arch).cuda();m.center.copy_(xc[fi].mean(0));m.scale.copy_(xc[fi].std(0,unbiased=False).clamp_min(.05));opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.1);logq=torch.zeros_like(priorw);saved={};history=[];best=(float('inf'),None)
          for step in range(1,401):
            m.train();shift=m(xc[fi]);z=base[fi,1]-base[fi,0]+shift # amount.5 in antisymmetric delta
            ce=F.binary_cross_entropy_with_logits(z,y[fi].float()*.97+.015,reduction='none');losses=ce.new_zeros(len(groups)).scatter_add_(0,gi,ce)/ng
            if obj=='dro':
                with torch.no_grad():logq.add_(.01*losses.detach());logq.sub_(logq.max());logq.clamp_(-8,0)
                gw=priorw*logq.exp();gw/=gw.sum()
            else:gw=priorw
            loss=(gw*losses).sum()+.05*shift.square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
            if step in (50,150,400):
                m.eval()
                with torch.no_grad():a=m(xc);z=binary_deploy(base,a,.5)
                nll=float((F.cross_entropy(z[ci],y[ci],reduction='none')*old.balance(d['rows'],d['labels'],cal).cuda()).sum()) if cal else None
                torch.save(dict(state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},architecture=arch,step=step,room=room,seed=seed,objective=obj,fit_identifiers=sorted({d['rows'][i]['identifier'] for i in fit})),dest/f'step{step}.pt');saved[step]=a.cpu();history.append(dict(step=step,calibration_nll=nll))
                if nll is not None and nll<best[0]:best=(nll,step)
          torch.save(saved,dest/'shifts.pt');save(dest/'complete.json',dict(history=history,selected_step=best[1],fit_windows=len(fit),calibration_windows=len(cal)));print('BINARY_FIT',room,arch,obj,seed,best[1],flush=True)
    results=[]
    for arch in ('linear','mlp'):
     for obj in ('balanced','dro'):
      for seed in (42,2026):
        steps=[read(WORK/'binary_fits'/f'{arch}_{obj}_{seed}_{r}'/'complete.json')['selected_step'] for r in ('a1','c1','g1','e1')];step=min((50,150,400),key=lambda k:abs(k-float(np.median(steps))));p=WORK/'binary_fits'/f'{arch}_{obj}_{seed}_actual'/'complete.json';v=read(p);v.update(selected_step=step,proxy_steps=steps);save(p,v)
      for seeds in ((42,),(2026,),(42,2026)):
       for amount in (.25,.5,1.):
        item=dict(architecture=arch,objective=obj,seeds=list(seeds),amount=amount,rooms={},method='relative_binary');net=0;improve=0;worst=1
        for room in ('a1','c1','g1','e1','actual'):
            d=torch.load(WORK/f'{room}.pt',weights_only=True);shifts=[]
            for seed in seeds:
                dest=WORK/'binary_fits'/f'{arch}_{obj}_{seed}_{room}';step=read(dest/'complete.json')['selected_step'];shifts.append(torch.load(dest/'shifts.pt',weights_only=True)[step])
            z=binary_deploy(d['base'],torch.stack(shifts).mean(0),amount);ix=old.split(room,d['rows'])[2];mt=old.metrics(z,d['labels'],d['base'],ix);item['rooms'][room]=mt
            if room!='actual':n=mt['corrected']-mt['broken'];net+=n;improve+=n>0;worst=min(worst,n/mt['total'])
        item.update(proxy_net=net,improving_rooms=improve,worst_room_delta=worst,eligible=net>0 and improve>=2 and worst>=-.02);results.append(item)
    save(WORK/'binary_results.json',results)
    for r in sorted(results,key=lambda v:(v['eligible'],v['proxy_net']),reverse=True)[:15]:print('BINARY_RESULT',r['architecture'],r['objective'],r['seeds'],r['amount'],r['proxy_net'],r['improving_rooms'],round(r['worst_room_delta'],3),r['rooms']['actual']['correct'],r['eligible'],flush=True)
if __name__=='__main__':main()
