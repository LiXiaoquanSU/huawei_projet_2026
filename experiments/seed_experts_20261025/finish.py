"""Whole-record uncertainty and error accounting; no export if the gate fails."""
import shutil
from run import *

def bootstrap(seed,method,rows):
    groups=[]
    for room in ROOMS:
        d=torch.load(WORK/f'pred_{room}_s{seed}_{method}.pt',weights_only=True)
        for identifier in sorted({rows[i]['identifier'] for i in d['indices']}):
            ix=[i for i in d['indices'] if rows[i]['identifier']==identifier]
            y=d['labels'][ix];p=d['logits'][ix].argmax(-1);q=d['baseline'][ix].argmax(-1)
            groups.append(dict(room=room,identifier=identifier,delta=int((p==y).sum()-(q==y).sum()),n=len(ix)))
    rng=np.random.default_rng(20261025);draws=[]
    # Stratify rooms, resample whole recordings within each, not correlated windows.
    for _ in range(5000):
        sampled=[]
        for room in ROOMS:
            rr=[r for r in groups if r['room']==room]
            sampled.extend(rr[i] for i in rng.integers(0,len(rr),size=len(rr)))
        draws.append(sum(r['delta'] for r in sampled)/sum(r['n'] for r in sampled))
    return dict(records=groups,ci95=[float(x) for x in np.percentile(draws,[2.5,97.5])],
                probability_nonpositive=float(np.mean(np.array(draws)<=0)),
                caveat='Development-set, after comparing10 fixed candidates. Not a fresh or selection-adjusted confidence guarantee.')

def main():
    rows=read(PREV/'rows.json');summary=read(WORK/'summary.json');rr=read(WORK/'results.json')
    for s in summary:
        s['bootstrap']=bootstrap(s['seed'],s['method'],rows)
        s['robust_pass']=s['passed'] and s['bootstrap']['ci95'][0]>0
        actual=next(r for r in rr if r['room']=='actual' and r['seed']==s['seed'] and r['method']==s['method'])
        s['source_diagnostic']=actual['metrics']
    save(WORK/'final_analysis.json',summary)
    passing=[s for s in summary if s['robust_pass']]
    selection=dict(selected=passing[0] if passing else None,status='ready_for_export' if passing else 'no_robust_candidate',
                   reason='Predeclared held-room gate plus stratified whole-recordbootstrap lower95%>0. Source score not selection criterion.',
                   platform_accuracy=None)
    save(WORK/'selection.json',selection)
    lines=['# 下一轮：跨种子条件专家实验结果','','目标平台0.60；当前已测最高仍0.54。本次没有老师隐藏标签。',
      '', '## 第一阶段：预设跨种子组合',
      '', '8次留房间从随机初始化训练；2个全量分享底座；10种固定组合（2种子×5规则）。校准不使用留出房间标签。',
      '代理房间已用于历史研究，且原代理权重来自历史训练，新代理采用本次数值管线；不是新的盲测或纯随机种子因果试验。',
      '', '|种子|融合方式|留房间总提升|a1|c1|g1|e1|整记录95%区间|通过|',
      '|---|---|---:|---:|---:|---:|---:|---|---|']
    for s in summary:
        gains=[f"{s['room_gains'][r]*100:+.2f}" for r in ROOMS]
        lo,hi=s['bootstrap']['ci95']
        lines.append('|'+ '|'.join([str(s['seed']),s['method'],f"{s['pooled_gain']*100:+.2f}",*gains,
                     f'{lo*100:+.2f}～{hi*100:+.2f}',str(s['robust_pass'])])+'|')
    lines+=['','单位：百分点；来源训练准确率仅保存在明细中，不当作隐藏评测成绩。',
      '', '第一阶段结论：'+('存在通过本地筛选的候选，仍需导出和资源/数值核验，平台提升未知。' if passing else
                 '没有候选通过预设筛选，不把表现不稳的融合另起名字当作升级模型。保留原0.54。'),
      '', '完整逐房间、混淆矩阵、纠错/破坏数和检查点在D:/huawei2026-experiments/seed_experts_20261025/。',
      '所有规则实现、校准及训练参数见本目录run.py；分享权重保持原样。']
    report='\n'.join(lines)+'\n'
    (ROOT/'docs/SEED_EXPERTS_20261025.md').write_text(report,encoding='utf-8')
    (PACKAGE/'下一轮实验_跨种子条件专家.md').write_text(report,encoding='utf-8')
    score_rows=[]
    for r in rr:
        m=r['metrics'];b=r['baseline']
        score_rows.append(dict(room=r['room'],seed=r['seed'],method=r['method'],
          split='训练来源诊断' if r['room']=='actual' else '历史开发留房间',accuracy=m['accuracy'],
          baseline_accuracy=b['accuracy'],corrected=m['corrected'],broken=m['broken'],
          changed=m['changed'],windows=m['total'],platform_score='未测试'))
    with (PACKAGE/'新实验_留房间结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(score_rows[0]));w.writeheader();w.writerows(score_rows)
    marker='\n## 9. 队友包到达后的下一轮：跨种子条件专家\n'
    full=ROOT/'docs/TEAM_PROGRESS_FULL_20261025.md';content=full.read_text(encoding='utf-8').split(marker)[0]
    best=summary[0]
    content+=marker+f'\n已完成8次留房间五轮训练，第一阶段比较seed42/310与原底座的5种组合。第一阶段最好开发汇总变化为{best["pooled_gain"]*100:+.2f}个百分点。'+('存在通过初筛及整记录bootstrap的候选，平台尚未测试。' if passing else '第一阶段没有组合通过预设稳定性门槛，保留原0.54，不冒称升级成功。')+'\n\n本轮配置、逐房间结果与限制见docs/SEED_EXPERTS_20261025.md；来包目录另附新实验_留房间结果.csv，不混入平台成绩。\n'
    full.write_text(content,encoding='utf-8');shutil.copy2(full,PACKAGE/'团队完整总结_含小全实验.md')
    readme=PACKAGE/'README.md';text=readme.read_text(encoding='utf-8').split('\n## 本轮实验进度\n')[0]
    text+='\n## 本轮实验进度\n\n8次留房间训练和10种预设组合已完成；详细见`下一轮实验_跨种子条件专家.md`和`新实验_留房间结果.csv`。这些是本地开发结果，不是新增平台分。\n'
    readme.write_text(text,encoding='utf-8')
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
