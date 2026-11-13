"""Stage2, motivated by seen c1 gains/g1 losses; explicitly reused development data."""
from run import *
from finish import bootstrap

NEW=('adjacent_up','adjacent_up_consensus','one_to_two')

def main():
    save(WORK/'ordinal_plan.json',dict(stage=2,methods=NEW,seeds=SEEDS,
      motivation='Stage1 c1 correction mainly1->2, g1 destructive1->0/4; restrict ordinal direction and distance.',
      leakage_note='Rules proposed after seeing a1/c1/g1 stage1 results, before inspecting e1. Reused development, not new blind evaluation.',
      thresholds='Same gate as stage1; no per-room special cases; unchanged logits except when allowed transition happens.'))
    rows=read(PREV/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows]);results=[]
    for room in (*ROOMS,'actual'):
        base_raw=torch.load(PREV/f'{room}.pt',weights_only=True)['raw'];bb,bt=old.calibration(room)
        base=old.policy(base_raw,bb,bt);ix=old.split(room,rows)[2]
        for seed in SEEDS:
            raw=torch.load(WORK/f'raw_{room}_s{seed}.pt',weights_only=True)['raw']
            c=read(WORK/f'cal_{room}_s{seed}.json');b=torch.tensor(c['bias']);T=c['temperature']
            for method in NEW:
                z=combine(base,raw,b,T,method,base_raw,bb,bt)
                results.append(dict(room=room,seed=seed,method=method,metrics=metric(z,base,y,ix,rows),baseline=metric(base,base,y,ix,rows)))
                torch.save(dict(logits=z,baseline=base,labels=y,indices=ix),WORK/f'pred_{room}_s{seed}_{method}.pt')
    summary=[]
    for seed in SEEDS:
        for method in NEW:
            rr=[r for r in results if r['seed']==seed and r['method']==method and r['room']!='actual']
            gains=[r['metrics']['accuracy']-r['baseline']['accuracy'] for r in rr]
            gain=sum(r['metrics']['correct']-r['baseline']['correct'] for r in rr)/sum(r['metrics']['total'] for r in rr)
            ci=bootstrap(seed,method,rows)
            passed=gain>=.02 and min(gains)>=-.03 and sum(g>0 for g in gains)>=2
            summary.append(dict(seed=seed,method=method,stage=2,pooled_gain=gain,room_gains=dict(zip(ROOMS,gains)),
               positive_rooms=sum(g>0 for g in gains),passed=passed,bootstrap=ci,robust_pass=passed and ci['ci95'][0]>0,
               source_diagnostic=next(r['metrics'] for r in results if r['room']=='actual' and r['seed']==seed and r['method']==method)))
    summary.sort(key=lambda r:(r['robust_pass'],r['pooled_gain']),reverse=True)
    save(WORK/'ordinal_results.json',results);save(WORK/'ordinal_summary.json',summary)
    passing=[s for s in summary if s['robust_pass']]
    if passing:
        save(WORK/'selection.json',dict(selected=passing[0],status='ready_for_export',platform_accuracy=None,
           reason='Stage2 bounded ordinal repair, conceived after stage1 seen rooms. Same numeric gate and bootstrap, development not blind.'))
    report=ROOT/'docs/SEED_EXPERTS_20261025.md'
    text=report.read_text(encoding='utf-8').split('\n## 第二阶段')[0]
    text+='\n## 第二阶段：限制人数修正方向和距离\n\n看到c1的1→2纠错、g1的1→0/4破坏后提出；提出时未查看e1结果。仍属于重复开发评估，不是新盲测。新增6种组合，不重新拟合基础权重或温度/偏置。\n\n'
    text+='|种子|方式|汇总提升/百分点|a1|c1|g1|e1|整记录95%区间|通过|\n|---|---|---:|---:|---:|---:|---:|---|---|\n'
    for s in summary:
        gains=[f'{s["room_gains"][r]*100:+.2f}' for r in ROOMS];lo,hi=s['bootstrap']['ci95']
        text+='|'+'|'.join([str(s['seed']),s['method'],f'{s["pooled_gain"]*100:+.2f}',*gains,f'{lo*100:+.2f}～{hi*100:+.2f}',str(s['robust_pass'])])+'|\n'
    selection=read(WORK/'selection.json')
    text+='\n最终状态：'+selection['status']+'。当前最高平台分仍0.54；尚未有本轮平台结果。\n'
    report.write_text(text,encoding='utf-8');(PACKAGE/'下一轮实验_跨种子条件专家.md').write_text(text,encoding='utf-8')
    csvpath=PACKAGE/'新实验_留房间结果.csv'
    with csvpath.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f));fields=list(rr[0])
    rr=[r for r in rr if r['method'] not in NEW]
    for r in results:
        m=r['metrics'];b=r['baseline']
        rr.append(dict(room=r['room'],seed=r['seed'],method=r['method'],split='训练来源诊断' if r['room']=='actual' else '历史开发留房间',
          accuracy=m['accuracy'],baseline_accuracy=b['accuracy'],corrected=m['corrected'],broken=m['broken'],changed=m['changed'],windows=m['total'],platform_score='未测试'))
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
    full=ROOT/'docs/TEAM_PROGRESS_FULL_20261025.md';content=full.read_text(encoding='utf-8')
    content=content.split('\n第二阶段补充：')[0]
    content+='\n第二阶段补充：增加6种相邻向上/1→2条件修正，避免g1出现1→0/4大跨度破坏。最终状态：'+selection['status']+'。详情见本轮实验报告；不能把数据驱动提出的第二阶段规则描述为全新盲测。\n'
    full.write_text(content,encoding='utf-8');(PACKAGE/'团队完整总结_含小全实验.md').write_text(content,encoding='utf-8')
    print(json.dumps([dict(seed=s['seed'],method=s['method'],gain=s['pooled_gain'],rooms=s['room_gains'],ci=s['bootstrap']['ci95'],passed=s['robust_pass']) for s in summary],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
