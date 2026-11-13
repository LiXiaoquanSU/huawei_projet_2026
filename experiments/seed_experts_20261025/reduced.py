"""Stage3 explicit accuracy/runtime tradeoff: reduce only the independent expert's views."""
from run import *
from finish import bootstrap

GROUPS={'rx2':[0],'rx4':[0,5],'rx6':[0,3,4]}

def main():
    save(WORK/'reduced_plan.json',dict(reason='Two12view models failed CPUtime: quick recheck~.80s >.7s; keep054 unchanged, reduce only expert views.',
       subsets=GROUPS,seed=310,base_rule='adjacent_up',calibration='refit per subset,150steps only calibration records',
       reused_development=True,no_new_training=True))
    rows=read(PREV/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows]);results=[]
    for group,indices in GROUPS.items():
        for room in (*ROOMS,'actual'):
            br=torch.load(PREV/f'{room}.pt',weights_only=True)['raw'];bb,bt=old.calibration(room);base=old.policy(br,bb,bt)
            raw=torch.load(WORK/f'raw_{room}_s310.pt',weights_only=True)['raw'][:,indices]
            fit,cal,test=old.split(room,rows);b,T=calibrate(raw,y,cal if room!='actual' else fit)
            save(WORK/f'cal_{room}_s310_{group}.json',dict(bias=b.tolist(),temperature=T,permutations=indices))
            method='adjacent_up_'+group;z=combine(base,raw,b,T,'adjacent_up',br,bb,bt)
            results.append(dict(room=room,seed=310,method=method,metrics=metric(z,base,y,test,rows),baseline=metric(base,base,y,test,rows)))
            torch.save(dict(logits=z,baseline=base,labels=y,indices=test),WORK/f'pred_{room}_s310_{method}.pt')
    summary=[]
    for group,indices in GROUPS.items():
        method='adjacent_up_'+group;rr=[r for r in results if r['method']==method and r['room']!='actual']
        gains=[r['metrics']['accuracy']-r['baseline']['accuracy'] for r in rr]
        gain=sum(r['metrics']['correct']-r['baseline']['correct'] for r in rr)/sum(r['metrics']['total'] for r in rr)
        ci=bootstrap(310,method,rows);passed=gain>=.02 and min(gains)>=-.03 and sum(g>0 for g in gains)>=2
        s=dict(seed=310,method='adjacent_up',group=group,permutations=indices,stage=3,pooled_gain=gain,
          room_gains=dict(zip(ROOMS,gains)),positive_rooms=sum(g>0 for g in gains),bootstrap=ci,
          passed=passed,robust_pass=passed and ci['ci95'][0]>0,
          exploratory_only=not(passed and ci['ci95'][0]>0),
          source_diagnostic=next(r['metrics'] for r in results if r['room']=='actual' and r['method']==method))
        s['exploratory_eligible']=gain>0 and min(gains)>=0 and s['positive_rooms']>=2 and ci['ci95'][0]>0
        summary.append(s)
    # Among exploratory schemes prioritize broader room gains and runtime headroom,
    # rather than a one-window advantage from more inference views.
    summary.sort(key=lambda s:(s['robust_pass'],s['exploratory_eligible'],s['positive_rooms'],-len(s['permutations']),s['pooled_gain']),reverse=True)
    eligible=[s for s in summary if s['robust_pass'] or s['exploratory_eligible']]
    save(WORK/'reduced_results.json',results);save(WORK/'reduced_summary.json',summary)
    save(WORK/'reduced_selection.json',dict(selected=eligible[0] if eligible else None,
      reason='Reduce runtime, rerun calibration per subset. Same formal gate; below2pp only explicitly exploratory. Among eligible exploratory choices prefer more improving rooms and fewer views, not a one-window accuracy advantage.',platform_score=None))
    p=ROOT/'docs/SEED_EXPERTS_20261025.md';text=p.read_text(encoding='utf-8').split('\n## 第三阶段')[0]
    text+='\n## 第三阶段：运行时间约束下减少第二专家视角\n\n双十二视角约0.80秒，未过0.7秒检查；该版本不交付。只改变第二模型，分别比较2/4/6视角，独立重新校准，原054十二视角与权重保持。属于后续开发探索，非全新盲测。\n\n|专家视角|汇总提升/百分点|a1|c1|g1|e1|95%区间|正式通过|\n|---|---:|---:|---:|---:|---:|---|---|\n'
    for s in summary:
        lo,hi=s['bootstrap']['ci95'];g=[f'{s["room_gains"][r]*100:+.2f}' for r in ROOMS]
        text+='|'+'|'.join([str(len(s['permutations'])*2),f'{s["pooled_gain"]*100:+.2f}',*g,f'{lo*100:+.2f}～{hi*100:+.2f}',str(s['robust_pass'])])+'|\n'
    text+='\n筛选结果：'+('有待导出核验的'+('探索候选，未达到正式门槛。' if eligible[0]['exploratory_only'] else '本地候选，平台未知。') if eligible else '未产生可交付候选。')+'\n'
    p.write_text(text,encoding='utf-8');(PACKAGE/'下一轮实验_跨种子条件专家.md').write_text(text,encoding='utf-8')
    print(json.dumps([dict(group=s['group'],gain=s['pooled_gain'],rooms=s['room_gains'],ci=s['bootstrap']['ci95'],eligible=s['exploratory_eligible'],robust=s['robust_pass']) for s in summary],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
