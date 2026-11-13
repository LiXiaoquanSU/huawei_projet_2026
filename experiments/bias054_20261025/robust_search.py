"""Bounded exhaustive4D bias grid, record-wise calibration recipe stability."""
import itertools,time
from evaluate_bias import *
RWORK=Path('D:/huawei2026-experiments/robust_bias054_20261025')

def grid():
    values=[round(i*.05,2) for i in range(-6,7)]
    return torch.tensor([[a,0.,b,c,d] for a,b,c,d in itertools.product(values,repeat=4)])

def cache(base,delta):
    path=RWORK/f'grid_{base}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    zs=[];ys=[];rows=[]
    for room in ('a1','c1','g1','e1'):
        z,y,rr=data(room,base);ix=old.split(room,rr)[2];zs.append(z[ix]);ys.append(y[ix]);rows += [rr[i] for i in ix]
    pz=torch.cat(zs);py=torch.cat(ys);sz,sy,srows=data('actual',base)
    def prepare(z,y,rr):
        records=sorted({r['identifier'] for r in rr});rec=[next(r for r in rr if r['identifier']==rid) for rid in records];ri=torch.tensor([records.index(r['identifier']) for r in rr]);membership=F.one_hot(ri,len(records)).float().cuda();return z.cuda(),y.cuda(),membership,rec,membership.sum(0)
    pp=prepare(pz,py,rows);ss=prepare(sz,sy,srows);out={k:[] for k in ('proxy_correct','proxy_nll','source_correct')};started=time.perf_counter()
    with torch.inference_mode():
      for start in range(0,len(delta),256):
        dd=delta[start:start+256].cuda()[:,None,:]
        for name,values in (('proxy',pp),('source',ss)):
            z,y,member,rec,n=values;logits=z[None]+dd;pred=logits.argmax(-1);correct=(pred==y[None]).float()@member;out[name+'_correct'].append(correct.cpu().short())
            if name=='proxy':
                nll=logits.logsumexp(-1)-logits.gather(-1,y[None,:,None].expand(len(dd),-1,1)).squeeze(-1);out['proxy_nll'].append(((nll@member)/n).cpu())
    d={k:torch.cat(v) for k,v in out.items()};d.update(proxy_records=pp[3],source_records=ss[3],proxy_sizes=pp[4].cpu(),source_sizes=ss[4].cpu(),delta=delta);torch.save(d,path);print('ROBUST_GRID',base,len(delta),round(time.perf_counter()-started,1),flush=True);return d

def fold_assignment(records,seed=20261025):
    # Keep each raw record intact. Balance folds within room/class when possible,
    # with room-fold sizes resolving groups that contain fewer than3 records.
    rng=np.random.default_rng(seed);out=np.full(len(records),-1);sizes={r:np.zeros(3,int) for r in ('a1','c1','g1','e1')}
    for room in sizes:
      for label in range(5):
        ii=[i for i,r in enumerate(records) if r['room']==room and r['headcount']==label];rng.shuffle(ii);counts=np.zeros(3,int)
        for i in ii:
            f=min(range(3),key=lambda f:(counts[f],sizes[room][f],f));out[i]=f;counts[f]+=1;sizes[room][f]+=1
    assert (out>=0).all();return torch.tensor(out)

def fit_select(d,indices,source_indices,radius,objective):
    delta=d['delta'];pc=d['proxy_correct'].float();pn=d['proxy_nll'];sc=d['source_correct'].float();control=int(delta.abs().sum(-1).argmin());eligible=delta.abs().amax(-1)<=radius+1e-6
    macros=[];nll=[];room_deltas=[]
    for room in ('a1','c1','g1','e1'):
        ix=[i for i in indices if d['proxy_records'][i]['room']==room]
        if not ix:continue
        a=pc[:,ix].sum(-1)/d['proxy_sizes'][ix].sum();macros.append(a);nll.append(pn[:,ix].mean(-1));room_deltas.append(a-a[control]);eligible &= a>=a[control]-.020001
    macro=torch.stack(macros).mean(0);nll=torch.stack(nll).mean(0);source=sc[:,source_indices].sum(-1);eligible &= source>=source[control]
    for label,budget in ((0,5),(4,1)):
        ii=[i for i in source_indices if d['source_records'][i]['headcount']==label]
        if ii:
            a=sc[:,ii].sum(-1);allowed=max(1,int(np.ceil(budget*float(d['source_sizes'][ii].sum())/(220 if label==0 else 140))))
            eligible &= a>=a[control]-allowed
    penalty=delta.square().mean(-1);score=macro-.02*penalty if objective=='accuracy' else -nll-.1*penalty
    score[~eligible]=-float('inf');best=score.max();tie=torch.nonzero(score>=best-1e-7).flatten();k=int(tie[delta[tie].abs().sum(-1).argmin()])
    return k,dict(feasible=int(eligible.sum()),macro=float(macro[k]),regularized_objective=float(score[k]))

def measurements(correct,sizes,records,base_correct):
    rooms={};macro=[];net=[]
    for room in ('a1','c1','g1','e1'):
        ix=[i for i,r in enumerate(records) if r['room']==room];n=int(sizes[ix].sum());c=int(correct[ix].sum());b=int(base_correct[ix].sum());rooms[room]=dict(correct=c,base_correct=b,total=n,net=c-b);macro.append(c/n);net.append(c-b)
    return dict(correct=int(correct.sum()),base_correct=int(base_correct.sum()),total=int(sizes.sum()),macro_accuracy=float(np.mean(macro)),net=int(correct.sum()-base_correct.sum()),rooms=rooms,improving_rooms=sum(v>0 for v in net),worst_room_delta=min(r['net']/r['total'] for r in rooms.values()))

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;RWORK.mkdir(parents=True,exist_ok=True);delta=grid();save(RWORK/'plan.json',dict(bases=list(BASES),grid=dict(step=.05,relative_to_class1=True,each_other_coordinate=[-.3,.3],vectors_per_base=len(delta),total_vectors=2*len(delta)),radii=[.1,.2,.3],objectives=['accuracy:equalroomaccuracy-.02meanbias2','nll:equalroomNLL+.1meanbias2'],record_folds=3,seed=20261025,source_guards='Only meta-fit records: totalcorrectnondecreasing,0personloss<=5 scaled,4personloss<=1 scaled; roomtrainingloss<=2pp.',selection='Compare12calibrationrecipesusingheld-recordabsoluteequalroomaccuracy,thenworstroom,thenabsoluteoverallcorrectness,thenradius; finalrefitonallrecordswithsameobjective. RequireCVnet>0,>=2improvingrooms,noworstloss>2pp. Sourceispreviouslytraineddiagnostic. Gridsearchpeaknotusedasindependentvalidation.',limitations='Predictionsfrom4alreadyusedouterroomteachers; teacherlevelfullynestedindependenceNOTclaimed,fullsourceteachersawlabels. Thisonlychecksrecordwisepostprocessselectionstability,notunseenblindrooms. Finalfitproxyresultsarecalibrationfit,notCV.'))
    recipes=[];fold_manifest={}
    for base in ('selective','gprime'):
      d=cache(base,delta);folds=fold_assignment(d['proxy_records']);fold_manifest[base]=[dict(identifier=r['identifier'],room=r['room'],headcount=r['headcount'],fold=int(folds[i])) for i,r in enumerate(d['proxy_records'])];control=int(delta.abs().sum(-1).argmin())
      for radius in (.1,.2,.3):
       for objective in ('accuracy','nll'):
        cv_correct=torch.zeros(len(folds));snapshots=[]
        for f in range(3):
            val=torch.nonzero(folds==f).flatten().tolist();fit=torch.nonzero(folds!=f).flatten().tolist();valids={d['proxy_records'][i]['identifier'] for i in val};sf=[i for i,r in enumerate(d['source_records']) if r['identifier'] not in valids]
            k,info=fit_select(d,fit,sf,radius,objective);cv_correct[val]=d['proxy_correct'][k,val].float();snapshots.append(dict(fold=f,delta=delta[k].tolist(),index=k,fit_records=len(fit),validation_records=len(val),**info))
        cv=measurements(cv_correct,d['proxy_sizes'],d['proxy_records'],d['proxy_correct'][control].float());k,fitinfo=fit_select(d,list(range(len(folds))),list(range(len(d['source_records']))),radius,objective);fitmetrics=measurements(d['proxy_correct'][k].float(),d['proxy_sizes'],d['proxy_records'],d['proxy_correct'][control].float())
        s=d['source_correct'][k].int();source=dict(correct=int(s.sum()),base_correct=int(d['source_correct'][control].sum()),classes=[int(s[[i for i,r in enumerate(d['source_records']) if r['headcount']==c]].sum()) for c in range(5)])
        item=dict(base=base,radius=radius,objective=objective,cv=cv,folds=snapshots,delta=[round(float(v),2) for v in delta[k]],index=k,fit=fitmetrics,source=source,final_fit_info=fitinfo,eligible=cv['net']>0 and cv['improving_rooms']>=2 and cv['worst_room_delta']>=-.020001 and fitmetrics['net']>0);recipes.append(item)
    save(RWORK/'folds.json',fold_manifest);save(RWORK/'recipes.json',recipes)
    ordered=sorted(recipes,key=lambda r:(r['eligible'],r['cv']['macro_accuracy'],r['cv']['worst_room_delta'],r['cv']['correct'],-r['radius']),reverse=True)
    for r in ordered:print('ROBUST_RECIPE',r['base'],r['radius'],r['objective'],'CV',r['cv']['correct'],r['cv']['net'],round(r['cv']['macro_accuracy'],5),'fit',r['fit']['correct'],'source',r['source']['correct'],r['delta'],r['eligible'],flush=True)
    if ordered[0]['eligible']:
        s=ordered[0];s['neighbors']=[];d=cache(s['base'],delta)
        for axis in (0,2,3,4):
            for shift in (-.05,.05):
                v=np.asarray(s['delta']);v[axis]=round(v[axis]+shift,2)
                if max(abs(v))>s['radius']+1e-6:continue
                k=int(((delta-torch.tensor(v,dtype=torch.float32)).abs().amax(-1)).argmin());s['neighbors'].append(dict(delta=v.tolist(),proxy_correct=int(d['proxy_correct'][k].sum()),source_correct=int(d['source_correct'][k].sum())))
        save(RWORK/'selection.json',s)
if __name__=='__main__':main()
