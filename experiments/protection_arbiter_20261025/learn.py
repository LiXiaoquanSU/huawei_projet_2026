"""Train relative-regret scorers on nested OOF examples; source-only selection."""
from arbiter_common import *

PROFILES=('linear','relation_mlp','count_mlp')
SEEDS=(42,2026)
ERROR_COST=8 if '--cost8' in sys.argv else 4 if '--cost4' in sys.argv else 2
if ERROR_COST!=2:PROFILES=('count_mlp',)
PAIR_ONLY='--pair-only' in sys.argv
PAIR_ONE='--pair-one' in sys.argv
AUX_HEADS='--aux-heads' in sys.argv
LABEL_CAL='--label-cal' in sys.argv
ALIGNED='--aligned' in sys.argv or LABEL_CAL
KEEP_ONE='--keep-one' in sys.argv
STRICT_SCOPE='--strict-scope' in sys.argv
ADJACENT_ONLY='--adjacent-only' in sys.argv
EVIDENCE_GUARD='--evidence-guard' in sys.argv or ADJACENT_ONLY
GUARD_ONLY='--guard-only' in sys.argv or PAIR_ONLY or PAIR_ONE or AUX_HEADS
OOF_ROOT=WORK/('aux_labelcal' if LABEL_CAL else 'aux_aligned' if ALIGNED else 'aux_heads') if AUX_HEADS else WORK
if AUX_HEADS:
    WORK=OOF_ROOT/'gates'
elif PAIR_ONE:
    WORK=WORK/'pair_one'
elif PAIR_ONLY:
    WORK=WORK/'pair_only'
elif GUARD_ONLY:
    WORK=WORK/'guard_only'
if ERROR_COST!=2:WORK=WORK/f'cost{ERROR_COST}'
FINAL_ROOT=WORK
REFINE_ONLY=(KEEP_ONE and not ALIGNED) or STRICT_SCOPE or EVIDENCE_GUARD
if EVIDENCE_GUARD:
    assert AUX_HEADS and LABEL_CAL and KEEP_ONE
    WORK=WORK/('adjacent_guard' if ADJACENT_ONLY else 'evidence_guard')
elif STRICT_SCOPE:
    assert AUX_HEADS and KEEP_ONE
    WORK=WORK/'strict_scope'
elif KEEP_ONE and not ALIGNED:
    assert AUX_HEADS
    WORK=WORK/'keep_one'

def original_one(prob):
    offset=prob.new_tensor([0,.35,0,0,0])
    return (prob[:,8].clamp_min(1e-12).log()-offset).argmax(-1)==1

def protection_mask(prob):
    offset=prob.new_tensor([0,.35,0,0,0])
    # Opinion8 is original input; opinion2 is its paired-time result.
    v0=(prob[:,8].clamp_min(1e-12).log()-offset).argmax(-1)
    v45=(prob[:,2].clamp_min(1e-12).log()-offset).argmax(-1)
    if PAIR_ONLY or PAIR_ONE or KEEP_ONE:return (v0!=1)&((v45==0)|(v45==2))
    return (v0==1)|(v45==0)|(v45==2)

def eligibility(prob,scope='views'):
    labels=prob.argmax(-1);eligible=(labels!=labels[:,:1])
    if AUX_HEADS:
        protected=protection_mask(prob)[:,None]
        newhead=torch.arange(prob.shape[1],device=prob.device)[None]>=20
        low=(labels[:,:1]<=1)&(labels<=1)&newhead
        if KEEP_ONE:low=low&~original_one(prob)[:,None]
        if STRICT_SCOPE or LABEL_CAL:low=low&False
        eligible=eligible&(protected|low)
    elif GUARD_ONLY:eligible=eligible&protection_mask(prob)[:,None]
    if PAIR_ONE:eligible=eligible&(labels==1)
    if EVIDENCE_GUARD:
        close=(labels-labels[:,:1]).abs()<=1
        stronger=prob.amax(-1)>=prob[:,0].amax(-1,keepdim=True)
        eligible=eligible&close
        if not ADJACENT_ONLY:eligible=eligible&stronger
    if scope=='group':eligible[:,2:]=False
    return eligible

def inputs(prob,profile):
    f=features(prob);n,k,_=f.shape
    typeid=torch.tensor([0,1]+[2]*6+[3]*12+(list(range(4,10)) if AUX_HEADS else []),device=f.device)
    typ=F.one_hot(typeid,num_classes=10 if AUX_HEADS else 4).to(f.dtype)[None].expand(n,-1,-1)
    x=torch.cat([f,f-f[:,:1],typ],-1)
    if profile=='count_mlp':
        label=F.one_hot(prob.argmax(-1),5).to(f.dtype)
        x=torch.cat([x,prob-prob[:,:1],label,label[:,:1].expand(-1,k,-1)],-1)
    return x

class Arbiter(nn.Module):
    def __init__(self,profile):
        super().__init__();self.profile=profile
        size=(39 if profile=='count_mlp' else 24)+(6 if AUX_HEADS else 0)
        self.net=nn.Linear(size,1) if profile=='linear' else nn.Sequential(nn.Linear(size,24),nn.SiLU(),nn.Linear(24,1))
        last=self.net if profile=='linear' else self.net[-1]
        nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    def forward(self,prob):
        out=self.net(inputs(prob,self.profile)).squeeze(-1)
        return out-out[:,:1]

def action(prob,regret,scope,threshold):
    eligible=eligibility(prob,scope)
    score=torch.where(eligible,regret,regret.new_full((),-100))
    best,index=score.max(-1)
    active=best>threshold
    index=torch.where(active,index,torch.zeros_like(index))
    return prob.gather(1,index[:,None,None].expand(-1,1,5)).squeeze(1).clamp_min(1e-12).log(),index

def weights(rows,y):
    groups={}
    for r,label in zip(rows,y):
        key=(int(label),r['identifier'] if r['room']=='unknown' else r['room']);groups[key]=groups.get(key,0)+1
    nc={c:sum(label==c for label,domain in groups) for c in range(5)}
    w=torch.tensor([1/(groups[(int(label),r['identifier'] if r['room']=='unknown' else r['room'])]*nc[int(label)]) for r,label in zip(rows,y)])
    return w/w.sum()

def train(prob,y,w,profile,seed,steps=300):
    torch.manual_seed(seed);model=Arbiter(profile)
    if GUARD_ONLY:
        mask=eligibility(prob).any(1) if AUX_HEADS else protection_mask(prob)
        w=w*mask.float()
        if float(w.sum())==0:return model.eval()
        w=w/w.sum()
    correct=(prob.argmax(-1)==y[:,None]).float()
    delta=correct-correct[:,:1]
    # A damaged originally correct prediction has twice the training cost.
    target=torch.where(delta<0,ERROR_COST*delta,delta)
    relevant=eligibility(prob).float()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=.05)
    for step in range(steps):
        score=model(prob);loss=(((score-target).square()*relevant).sum(-1)/relevant.sum(-1).clamp_min(1)*w).sum()
        loss=loss+.01*score.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    return model.eval()

def source_selection(room):
    if (WORK/f'source_selection_{room}.json').exists():return read(WORK/f'source_selection_{room}.json')['selected']
    d=torch.load(OOF_ROOT/f'oof_{room}.pt',weights_only=True);p=d['probabilities'];y=d['labels'];w=weights(d['rows'],y)
    labels=p.argmax(-1);baseline=labels[:,0]==y;folds=torch.tensor(d['folds']);candidates=[]
    for profile in PROFILES:
        out=torch.zeros_like(p[:,:,0]);states=[]
        if REFINE_ONLY:
            previous=torch.load(FINAL_ROOT/f'selection_{room}_{profile}.pt',weights_only=True)
            out=previous['source_oof_regret'];states=previous['states']
        else:
            for fold in range(3):
                fit=folds!=fold;val=~fit
                ww=w[fit];ww/=ww.sum()
                for seed in SEEDS:
                    model=train(p[fit],y[fit],ww,profile,seed)
                    with torch.no_grad():out[val]+=model(p[val])/len(SEEDS)
                    states.append(dict(fold=fold,seed=seed,state=model.state_dict()))
        choices=[]
        for scope in ['group','views']:
            for threshold in [0.,.1,.2,.35,.5]:
                z,ix=action(p,out,scope,threshold);right=z.argmax(-1)==y
                gain=float(((right.float()-baseline.float())*w).sum())
                corrected=int((right&~baseline).sum());broken=int((~right&baseline).sum())
                choices.append(dict(profile=profile,scope=scope,threshold=threshold,balanced_gain=gain,corrected=corrected,broken=broken,
                                    net=corrected-broken,changes=int((ix!=0).sum())))
        # Source-only selection; default abstention wins if all candidates fail.
        choices.append(dict(profile=profile,scope='views',threshold=100.,balanced_gain=0.,corrected=0,broken=0,net=0,changes=0))
        best=max(choices,key=lambda c:(c['balanced_gain'],-c['changes'],c['threshold']))
        candidates.append(best)
        torch.save(dict(profile=profile,states=states,source_oof_regret=out,choices=choices,selected=best),WORK/f'selection_{room}_{profile}.pt')
        print('ARBITER_SOURCE_SELECT',room,best,flush=True)
    selected=max(candidates,key=lambda c:(c['balanced_gain'],-c['changes']))
    save(WORK/f'source_selection_{room}.json',dict(selected=selected,candidates=candidates,selection_labels='Only source record-OOF; outer room excluded from every teacher'))
    return selected

def final_models(room,profile):
    path=FINAL_ROOT/f'arbiters_{room}_{profile}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    d=torch.load(OOF_ROOT/f'oof_{room}.pt',weights_only=True);p=d['probabilities'];y=d['labels'];w=weights(d['rows'],y)
    states=[]
    for seed in SEEDS:
        model=train(p,y,w,profile,seed);states.append(dict(seed=seed,state=model.state_dict()))
    torch.save(dict(profile=profile,states=states,source_records=sorted({r['identifier'] for r in d['rows']})),path)
    return torch.load(path,weights_only=True)

def apply(prob,cp,config):
    outputs=[]
    with torch.no_grad():
        for s in cp['states']:
            model=Arbiter(cp['profile']);model.load_state_dict(s['state']);model.eval();outputs.append(model(prob))
    regret=torch.stack(outputs).mean(0)
    return action(prob,regret,config['scope'],config['threshold'])

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    rooms=(*ROOMS,'actual') if '--include-actual' in sys.argv else ['actual'] if '--actual' in sys.argv else ROOMS
    for room in rooms:
        selected=source_selection(room)
        for profile in PROFILES:final_models(room,profile)
    save(WORK/'gate_plan.json',dict(guard_only=GUARD_ONLY,pair_only=PAIR_ONLY,pair_one=PAIR_ONE,source_oof_root=str(OOF_ROOT),error_cost=ERROR_COST,training=f'Sharedrelativeutility; falsechange cost{ERROR_COST}; only protectedexamples ifguardonly.',
                                  selection='SourceOOF record folds, neverouter target labels. Guard-only revision after broad arbitration failed. Reused development.'))
    save(WORK/('actual_gate_complete.json' if '--actual' in sys.argv else 'proxy_gate_complete.json'),dict(rooms=list(rooms)))

if __name__=='__main__':main()
