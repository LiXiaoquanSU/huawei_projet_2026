"""Prior-free view log-odds evidence rather than calibrated confidence transfer."""
from arbiter_common import *
from learn import weights
WORK=WORK/'prior_free'

def data(raw,b,T):
    p=opinions(raw,b,T);offset=raw.new_tensor([0,.35,0,0,0])
    v0=(raw[:,0,0]/T+b).argmax(-1)
    pair=raw[:,0].exp().mean(1).clamp_min(1e-12).log()
    v45=(pair/T+b).argmax(-1)
    active=(v0!=1)&((v45==0)|(v45==2))
    groups=raw.exp().mean(2).clamp_min(1e-12).log();pool=raw.exp().mean((1,2)).clamp_min(1e-12).log()
    # Active rows always use timepair; other rows cannot be changed.
    u=torch.cat([pair[:,None],pool[:,None],groups,raw.flatten(1,2)],1)
    c=p.argmax(-1);base=c[:,:1];g=lambda z,ix:z.gather(-1,ix.unsqueeze(-1)).squeeze(-1)
    uc=g(u,c);ub=g(u,base.expand_as(c));bc=g(u[:,:1].expand_as(u),c);bb=g(u[:,:1],base)
    alt=uc-ub;original=bc-bb;gain=alt-original
    support=(groups[:,:,None,:].expand(-1,-1,20,-1).gather(-1,c[:,None,:,None].expand(-1,6,-1,1)).squeeze(-1)>
             groups[:,:,None,:].expand(-1,-1,20,-1).gather(-1,base[:,None,:,None].expand(-1,6,20,1)).squeeze(-1)).float().mean(1)
    pu=u.softmax(-1);entropy=-(pu*pu.clamp_min(1e-10).log()).sum(-1)/np.log(5)
    f=torch.stack([alt.clamp(-8,8)/4,original.clamp(-8,8)/4,gain.clamp(-8,8)/4,
                   support,entropy,entropy-entropy[:,:1],g(pu,c),g(pu,base.expand_as(c))],-1)
    typ=F.one_hot(torch.tensor([0,1]+[2]*6+[3]*12,device=raw.device),4).float()[None].expand(len(raw),-1,-1)
    f=torch.cat([f,typ],-1)
    eligible=active[:,None]&(c!=base)&((c-base).abs()<=1)&(alt>0)
    return p,f,eligible

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(12,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,f):return self.net(f).squeeze(-1)

def fit(f,p,e,y,w,seed):
    torch.manual_seed(seed);m=Gate();opt=torch.optim.AdamW(m.parameters(),lr=.005,weight_decay=.1)
    right=(p.argmax(-1)==y[:,None]).float();delta=right-right[:,:1];target=torch.where(delta<0,4*delta,delta)
    for _ in range(300):
        out=m(f);loss=(((out-target).square()*e).sum(-1)/e.sum(-1).clamp_min(1)*w).sum()+.01*out.square().mean()
        opt.zero_grad();loss.backward();opt.step()
    return m.eval()

def choose(p,s,e,threshold,amount):
    score=torch.where(e,s,s.new_full((),-100));v,ix=score.max(-1);ix=torch.where(v>threshold,ix,torch.zeros_like(ix))
    alt=p.gather(1,ix[:,None,None].expand(-1,1,5)).squeeze(1)
    return ((1-amount)*p[:,0]+amount*alt).clamp_min(1e-12).log(),ix

def oof(room):
    original=WORK.parent;parts=[];features_all=[];eligible=[]
    for k in range(3):
        d=torch.load(original/'crossfit'/room/f'fold{k}/predictions.pt',weights_only=True)
        p,f,e=data(d['raw'],d['bias'],d['temperature']);parts.append(p);features_all.append(f);eligible.append(e)
    d=torch.load(original/f'oof_{room}.pt',weights_only=True)
    return d,torch.cat(parts),torch.cat(features_all),torch.cat(eligible)

def train_room(room):
    path=WORK/f'{room}.pt'
    if path.exists():return
    d,p,f,e=oof(room);y=d['labels'];w=weights(d['rows'],y);folds=torch.tensor(d['folds']);score=torch.zeros(p.shape[:2])
    for k in range(3):
        ii=folds!=k;jj=~ii;ww=w[ii];ww/=ww.sum()
        for seed in [42,2026]:
            m=fit(f[ii],p[ii],e[ii],y[ii],ww,seed)
            with torch.no_grad():score[jj]+=m(f[jj])/2
    nll=lambda z:float((-z.gather(1,y[:,None]).squeeze(1)*w).sum())
    choices=[]
    for threshold in [0.,.1,.25,.5]:
        for amount in [.25,.5,1.]:
            z,ix=choose(p,score,e,threshold,amount);choices.append(dict(threshold=threshold,amount=amount,nll=nll(z)))
    choices.append(dict(threshold=100.,amount=0.,nll=nll(p[:,0].clamp_min(1e-12).log())))
    config=min(choices,key=lambda c:(round(c['nll'],7),c['amount'],-c['threshold']));states=[]
    for seed in [42,2026]:states.append(fit(f,p,e,y,w,seed).state_dict())
    torch.save(dict(config=config,states=states,choices=choices),path);print('PRIOR_FREE_SOURCE',room,config,flush=True)

def apply(raw,b,T,cp):
    p,f,e=data(raw,b,T);scores=[]
    with torch.no_grad():
        for state in cp['states']:
            m=Gate();m.load_state_dict(state);scores.append(m(f))
    return choose(p,torch.stack(scores).mean(0),e,cp['config']['threshold'],cp['config']['amount'])

def main():
    torch.set_num_threads(4);WORK.mkdir(exist_ok=True)
    for room in (*ROOMS,'actual'):train_room(room)
    spec=importlib.util.spec_from_file_location('prior_free_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    r=dict(rooms={},records=[])
    for room in (*ROOMS,'actual'):
        d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);cp=torch.load(WORK/f'{room}.pt',weights_only=True)
        z,a=apply(d['raw'],b,T,cp);base=old.policy(d['raw'],b,T);ix=old.split(room,d['rows'])[2];m=old.metrics(z,d['labels'],base,ix)
        torch.save(dict(logits=z,actions=a,baseline=base,labels=d['labels'],indices=ix),WORK/f'eval_{room}.pt')
        if room=='actual':save(WORK/'actual_metrics.json',m);continue
        r['rooms'][room]=m
        for identifier in sorted({d['rows'][i]['identifier'] for i in ix}):
            jj=[i for i in ix if d['rows'][i]['identifier']==identifier];yy=d['labels'][jj]
            c=int((z[jj].argmax(-1)==yy).sum());bc=int((base[jj].argmax(-1)==yy).sum())
            r['records'].append(dict(room=room,identifier=identifier,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
    r['net']=sum(x['delta'] for x in r['records']);r['ci95']=ev.bootstrap(r['records']);r['room_gains']={room:sum(x['delta'] for x in r['records'] if x['room']==room)/r['rooms'][room]['total'] for room in ROOMS}
    r['passed']=r['net']>=18 and r['ci95'][0]>0 and min(r['room_gains'].values())>=-.02 and sum(g>0 for g in r['room_gains'].values())>=2
    save(WORK/'results.json',r);print(json.dumps(dict(net=r['net'],ci=r['ci95'],rooms=r['room_gains'],actual=read(WORK/'actual_metrics.json')['correct'],passed=r['passed']),indent=2),flush=True)

if __name__=='__main__':main()
