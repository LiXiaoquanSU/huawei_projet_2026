"""Check split sensitivity before preferring tiny apparent improvements."""
from robust_search import *
def main():
    torch.set_num_threads(4);seeds=[17,137,2026,90421,314159]
    save(RWORK/'repeat_plan.json',dict(reason='Initialsingle3foldsplit:radius.3and.2selective recipesdifferedbyonlyonecorrectwindow. Assessadditionalfixedsplitsbeforesayingwhichbest.',seeds=seeds,recipes=12,choice='RequiremeanCVgain>0,>=4of5positive repeats,>=2roomsmeanimproved,noworstmeanroomdrop>2pp. Rankabsoluteequalroommacroaccuracy. Ifrecipeswithin0.25percentagepointsoftopmean,prefer smallerradius,thenlowersplitstd,thenhighermean. This amendmentfollowsinitialresults; repeatsreuseidenticaldata,notnewindependentconfirmation.',final='Samefullfitrecipe,sourceguards unchanged. Noadditionalunboundedgrid.'))
    result=[]
    for base in ('selective','gprime'):
      d=torch.load(RWORK/f'grid_{base}.pt',weights_only=True);control=int(d['delta'].abs().sum(-1).argmin())
      for radius in (.1,.2,.3):
       for objective in ('accuracy','nll'):
        repeats=[]
        for seed in seeds:
            folds=fold_assignment(d['proxy_records'],seed);cv_correct=torch.zeros(len(folds));snapshots=[]
            for f in range(3):
                val=torch.nonzero(folds==f).flatten().tolist();fit=torch.nonzero(folds!=f).flatten().tolist();ids={d['proxy_records'][i]['identifier'] for i in val};sf=[i for i,r in enumerate(d['source_records']) if r['identifier'] not in ids];k,info=fit_select(d,fit,sf,radius,objective);cv_correct[val]=d['proxy_correct'][k,val].float();snapshots.append(dict(fold=f,delta=d['delta'][k].tolist(),index=k,**info))
            cv=measurements(cv_correct,d['proxy_sizes'],d['proxy_records'],d['proxy_correct'][control].float());repeats.append(dict(seed=seed,cv=cv,folds=snapshots))
        macros=[r['cv']['macro_accuracy'] for r in repeats];nets=[r['cv']['net'] for r in repeats];roomnet={room:float(np.mean([r['cv']['rooms'][room]['net'] for r in repeats])) for room in ('a1','c1','g1','e1')};roomn={room:repeats[0]['cv']['rooms'][room]['total'] for room in roomnet}
        k,info=fit_select(d,list(range(len(d['proxy_records']))),list(range(len(d['source_records']))),radius,objective);fitmetrics=measurements(d['proxy_correct'][k].float(),d['proxy_sizes'],d['proxy_records'],d['proxy_correct'][control].float());sc=d['source_correct'][k].int()
        item=dict(base=base,radius=radius,objective=objective,delta=[round(float(v),2) for v in d['delta'][k]],index=k,mean_cv_macro=float(np.mean(macros)),std_cv_macro=float(np.std(macros)),mean_cv_correct=float(np.mean([r['cv']['correct'] for r in repeats])),mean_cv_net=float(np.mean(nets)),positive_repeats=sum(n>0 for n in nets),mean_room_net=roomnet,repeats=repeats,fit=fitmetrics,source=dict(correct=int(sc.sum()),base_correct=int(d['source_correct'][control].sum()),classes=[int(sc[[i for i,r in enumerate(d['source_records']) if r['headcount']==c]].sum()) for c in range(5)]),eligible=np.mean(nets)>0 and sum(n>0 for n in nets)>=4 and sum(n>0 for n in roomnet.values())>=2 and min(roomnet[r]/roomn[r] for r in roomnet)>=-.020001 and fitmetrics['net']>0);result.append(item)
    for r in result:r['eligible']=bool(r['eligible'])
    save(RWORK/'repeat_results.json',result);eligible=[r for r in result if r['eligible']];assert eligible,'No stablecalibrationrecipe'
    best=max(r['mean_cv_macro'] for r in eligible);close=[r for r in eligible if r['mean_cv_macro']>=best-.0025];winner=min(close,key=lambda r:(r['radius'],r['std_cv_macro'],-r['mean_cv_macro']))
    winner['choice']=dict(best_mean_macro=best,tolerance=.0025,near_best_recipes=[dict(base=r['base'],radius=r['radius'],objective=r['objective'],mean_cv_macro=r['mean_cv_macro']) for r in close],method='Prefer smallerboundedchange among nearbest repeatedrecordCV recipes, notsingle gridmaximum.')
    d=torch.load(RWORK/f"grid_{winner['base']}.pt",weights_only=True);winner['neighbors']=[]
    for axis in (0,2,3,4):
      for shift in (-.05,.05):
        v=np.asarray(winner['delta']);v[axis]=round(v[axis]+shift,2)
        if max(abs(v))>winner['radius']+1e-6:continue
        k=int((d['delta']-torch.tensor(v,dtype=torch.float32)).abs().amax(-1).argmin());winner['neighbors'].append(dict(delta=v.tolist(),proxy_correct=int(d['proxy_correct'][k].sum()),source_correct=int(d['source_correct'][k].sum())))
    save(RWORK/'selection_repeated.json',winner)
    for r in sorted(result,key=lambda x:x['mean_cv_macro'],reverse=True):print('REPEATED_BIAS',r['base'],r['radius'],r['objective'],round(r['mean_cv_macro'],6),round(r['std_cv_macro'],6),r['mean_cv_correct'],r['positive_repeats'],r['delta'],r['eligible'],flush=True)
    print('REPEATED_SELECTED',winner['base'],winner['radius'],winner['delta'],winner['source'],flush=True)
if __name__=='__main__':main()
