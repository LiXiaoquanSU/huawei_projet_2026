"""Persist both positive and negative architecture results without changing scores."""
import csv
from pathlib import Path
import numpy as np
from train import ROOT,read,save,sha

def main():
    lines=['# 两个0.39结构之后的研究结果','', '平台已确认：cross_events seed42与context_events seed2026均0.39。本报告中的准确率都是反复使用的本地整房间开发结果，不是平台新分数。','']
    all_fits=0;exports=[]
    for tag in ('evidence_20261021','query_20261021'):
        work=Path('D:/huawei2026-experiments')/tag
        if not (work/'selection.json').exists():continue
        s=read(work/'selection.json');fits=list((work/'fits').glob('*/metrics.json'));all_fits+=len(fits)
        for path in fits:
            m=read(path);assert m['config']['external_data'] is False and m['config']['epochs']==5
            with (path.parent/'train_manifest.csv').open(encoding='utf-8-sig') as stream:training=list(csv.DictReader(stream))
            if m['config']['fold']=='full':assert m['actual_records']==len({r['identifier'] for r in training})==48
            else:
                with (path.parent/'validation_manifest.csv').open(encoding='utf-8-sig') as stream:validation=list(csv.DictReader(stream))
                assert {r['identifier'] for r in training}.isdisjoint({r['identifier'] for r in validation})
                assert all(r['room'] not in (m['config']['fold'],'unknown') for r in training)
        lines += [f'## {tag}',f'完整拟合{len(fits)}次；按预设本地选择指标超过原0.39：{s["improved_development"]}。','', '|结构|种子|房间均值|宏召回|最差房间|','|---|---:|---:|---:|---:|']
        results=s['controls']+s['screen']+[r for c in s['candidates'] for r in c['results'] if r['seed']==2026]
        for r in results:lines.append(f"|{r['architecture']}|{r['seed']}|{r['room_mean']:.2%}|{r['macro_recall']:.2%}|{r['worst']:.2%}|")
        for c in s['candidates']:
            cm=np.asarray([r['confusion_matrix'] for r in c['results']]).sum(0)
            lines += ['',f"{c['architecture']} 两种子房间均值{c['two_seed_room_mean']:.2%}；0—4类召回："+', '.join(f'{v:.1%}' for v in np.diag(cm)/cm.sum(1))+f"。seed42 e1={c['e1']['accuracy']:.2%}。"]
        if (work/'export.json').exists():
            e=read(work/'export.json');exports.append(e);lines+=['',f"候选：{e['path']}，SHA256 {e['sha256']}；平台未测。"]
        lines+=['']
    ordinal=read(ROOT/'checks/ordinal_readout_20261021.json');exports.append(ordinal)
    lines+=['## 输出端诊断与优先候选','固定三种读出对照发现，两个结构/两个种子四组中，已有序数概率单独输出均优于原分类/序数等权平均。context两种子房间均值28.66%→31.77%，cross29.24%→30.85%。采用原已获0.39的context seed2026完整权重，改变最终读出，不新增训练。没有调融合系数。',f"建议优先测试：{ordinal['alias']}；SHA256 {ordinal['sha256']}，{ordinal['bytes']}字节。它的预测会不同于原上下文模型，平台分数必须留空。",'仍有明显局限：本地4人召回0，e1 seed42仍8.33%；局部均值提高不是全面提高。两种子与反复使用的开发房间不能证明统计显著。','两个原0.39模型固定等权概率融合的配对种子平均房间准确率约28.84%，没有超过cross对照，未导出该融合。', '']
    lines+=['## 审计与限制',f'共{all_fits}次完整新拟合，不含一次因修正未启动timed模块dropout而中断的aligned/g1四轮训练；该历史已保存在D盘。原基线三折结果精确复现。预检查覆盖天线对方向、频率与片段身份、时间平移不变性及实际间隔敏感性。','两种子与少量房间不足以证明泛化提升；官方隐藏类比例未知。开发反复使用存在选择过拟合。所有新训练只用官方有标签数据，无外部/U/全量旧权重初始化。最终全量覆盖48记录；采样窗口覆盖以各metrics为准。','研究没有设置100MB参数筛选；交付仍必须遵守模型尺寸、CPU时间和内存要求。两份已评测0.39仍是并列平台最佳。','源码见experiments/evidence_20261021；计划PLAN.md，研究前分析docs/PHYSICS_STRUCTURE_AFTER_039.md。']
    (ROOT/'docs/AFTER_039_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for e in exports:assert sha(ROOT/e['path'])==sha(ROOT/e['alias'])==e['sha256'] and e['platform_score'] is None
    assert sha(ROOT/'models/best_039/model.onnx')=='5553bdec10f5faeab55759aab5750c522634ae149a4730bbe185cf26710ea24d'
    assert sha(ROOT/'models/best_039_context/model.onnx')=='f2ed6fb013e74cdf8d22f3597d11250df517f73a852048603efa64993b50cfb8'
    save(ROOT/'checks/after_039_research.json',dict(complete_fits=all_fits,exports=[{k:e[k] for k in ('path','sha256','platform_score','bytes')} for e in exports],platform_best=.39,record_disjoint_verified=True,full48_coverage_verified=True,both_platform039_hashes_unchanged=True))
    registry=read(ROOT/'models/registry.json');registry['recommendations']['next_submission']='ordinal_readout_20261021';registry['recommendations']['next_submission_note']='fixed ordinal readout; local development improvement only; both platform039 models remain best';save(ROOT/'models/registry.json',registry)
    print('SUMMARY',all_fits,[(e['architecture'],e['development']['two_seed_room_mean']) for e in exports])
if __name__=='__main__':main()
