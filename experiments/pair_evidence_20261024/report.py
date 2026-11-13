"""Publish one explicitly exploratory candidate after raw ONNX/resource checks."""
import csv
import json
import shutil
from pathlib import Path
from build import ROOT,WORK,EXPORT,SOURCE,SOURCE_HASH,sha,save,read


def main():
    published=ROOT/'models/temporal_rescue_20261024/01_temporal_consensus80.json'
    if published.exists() and read(published).get('platform_score') is not None:
        print('Scored artifact already published; preserve feedback and later candidate selection.');return
    check=read(EXPORT/'resource_check.json');validation=read(EXPORT/'validation.json')
    digest=sha(EXPORT/'model.onnx')
    assert check['status']=='pass' and check['sha256']==validation['candidate_sha256']==digest
    assert sha(SOURCE)==SOURCE_HASH
    # Raw source provenance remains identical to the prior actual-model audit.
    relevant={r['identifier'] for r in read(EXPORT/'all_predictions.json') if r['prediction']==0}
    sources=read(Path('D:/huawei2026-experiments/actual047_audit_20261024/summary.json'))['source_files']
    verified=[]
    for r in sources:
        if r['identifier'] not in relevant:continue
        for prefix,key in [('CSI','csi_sha256'),('ATI','ati_sha256')]:
            path=Path('D:/huawei2026-data/labeled')/f"{prefix}_{r['identifier']}.b2"
            assert sha(path)==r[key]
        verified.append(r)
    assert len(verified)==len(relevant)
    save(EXPORT/'raw_source_proof.json',dict(verified_records=len(verified),source_files=verified))
    destination=ROOT/'models/temporal_rescue_20261024';destination.mkdir(parents=True,exist_ok=True)
    target=destination/'01_temporal_consensus80.onnx'
    if target.exists():assert sha(target)==digest
    else:shutil.copy2(EXPORT/'model.onnx',target)
    metadata=dict(id='temporal_consensus80_20261024',path=target.relative_to(ROOT).as_posix(),sha256=digest,
        bytes=target.stat().st_size,platform_score=None,status='exploratory_awaiting_platform',
        base_platform_score=.48,base_sha256=SOURCE_HASH,report='docs/PAIR_EVIDENCE_20261024.md',
        method='base0 only; original+time-reversed pre-bias argmax1 AND both log(p1/p0)>=log4; otherwise preserve exact048 output',
        caveat='Source-data diagnostics, not independent test. Allow modest empty false positives as an explicit exploratory tradeoff; do not promise .54.',
        local_validation=validation,resource_check=check)
    save(destination/'01_temporal_consensus80.json',metadata)
    results=read(WORK/'results.json');assert len(results)==32
    training=[]
    for origin in ('actual_diagnostic','independent_proxy'):
        for pair in (False,True):
            for seed in (42,2026):
                group=[r for r in results if r['origin']==origin and r['pair_loss']==pair and r['seed']==seed]
                assert len(group)==4
                training.append(dict(origin=origin,pair_loss=pair,seed=seed,
                    corrected=sum(r['metrics']['corrected'] for r in group),broken=sum(r['metrics']['broken'] for r in group),
                    net_gain_rooms=[r['room'] for r in group if r['metrics']['corrected']>r['metrics']['broken']]))
    summary=dict(candidate=metadata,training_comparison=training,guards=read(WORK/'guard_diagnostics.json'))
    save(WORK/'summary.json',summary);save(ROOT/'checks/pair_evidence_20261024.json',summary)
    registry=read(ROOT/'models/registry.json');assert registry['current_best']['platform_score']==.48
    registry['research_candidates']=[r for r in registry.get('research_candidates',[]) if r.get('id')!=metadata['id']]+[metadata]
    registry['next_submission']={k:metadata[k] for k in ('id','path','sha256','platform_score','status','report')}
    registry['latest_research']=dict(report=metadata['report'],status='completed_exploratory_candidate',best_scored_unchanged=.48)
    registry['research_status']='temporal_consensus80_awaiting_platform'
    save(ROOT/'models/registry.json',registry)
    csv_path=ROOT/'models/platform_scores.csv'
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;score_rows=list(reader)
    if not any(r['sha256']==digest for r in score_rows):
        score_rows.append(dict(model='双时间视角去偏置，80%条件证据保护',platform_score='',path=metadata['path'],sha256=digest,
            source='local research candidate 2026-10-24; not platform evaluated',status='exploratory_awaiting_platform',
            submission_id='',availability='local',original_path=metadata['path'],archive_manifest=''))
        with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(score_rows)
    guards=read(WORK/'guard_diagnostics.json')
    lines=['# 成对证据训练与双时间视角去偏置保护','',
        '**平台最佳仍为0.48。** 本轮导出一份未评测的探索候选，不声称已达到0.54。',
        '',f"候选：`{metadata['path']}`，{metadata['bytes']/1e6:.2f}MB，SHA256 `{digest}`。",'',
        '## 本轮实验','',
        '第一组：去掉原始强度/时间摘要和高维房间特征，仅保留16个有方向的证据：4视角×直接头0/1对数比、序数头0/1对数比、静态专家0/1对数比、presence对数几率。所有权重为非负并归一化，只有17个参数。',
        '','比较普通类别/房间平衡二分类，与额外同房间0/1记录均值的成对排序损失（权重0.5）。固定600步、阈值0.8；4个外层留房间、2个新读出种子、2个损失、实际/独立代理两组证据，共32次训练。实际编码器见过全部标签，仅作诊断；独立代理只在其自己的训练侧拟合。此前固定校准记录仍从fit排除，没有用外层选阈值。',
        '','| 证据来源 | 成对损失 | 种子 | 纠正/改坏 | 净改善房间 |','|---|---|---:|---:|---|']
    for r in training:lines.append(f"| {r['origin']} | {'有' if r['pair_loss'] else '无'} | {r['seed']} | {r['corrected']}/{r['broken']} | {','.join(r['net_gain_rooms']) or '无'} |")
    full0=read(WORK/'full_data/pair0/diagnostic.json')['metrics'];full1=read(WORK/'full_data/pair1/diagnostic.json')['metrics']
    lines+=['',f"另用全部真实0/1标签拟合两个实际读出：普通版纠正{full0['corrected']}/改坏{full0['broken']}，成对版纠正{full1['corrected']}/改坏{full1['broken']}。属于训练集拟合诊断；本轮没有把它们导出为提交模型。成对损失尚无稳定跨房间优势，不能因全数据表面较好而认定成功。",'',
        '## 第二组：不训练新权重，改变何时使用去偏置证据','',
        '原0.48保留。仅当它预测0人时，检查原视角、正确时间反转两路的原网络去偏置输出；两路均须在5类中选择1人。候选再要求两路各自的1人分数/0人分数≥4，即只在0/1内归一化时1人占比≥80%。这不是校准后的五分类80%置信度，也不保证静止人检测能力。',
        '','| 缓存诊断规则 | 纠正 | 改坏 | 正确/960 |','|---|---:|---:|---:|']
    for name,r in guards.items():lines.append(f"| {name} | {r['corrected']} | {r['broken']} | {r['correct']} |")
    lines+=['','80%是本轮在宽松一致性规则之后检查的一个固定保守门槛，没有遍历阈值网格；规则仍来源于已反复使用的开发数据，不能视为独立验证。选择它是平台探索中的收益/风险权衡，并非满足上一轮空房损失≤1%的标准。',
        '','## 实际候选验证','',
        f"原始CSI直接运行全部{validation['raw_directly_checked']}个底座预测0的窗口，候选实际纠正{validation['corrected']}、改坏{validation['broken']}，总正确{validation['correct']}/960（{validation['accuracy']:.3%}）。其余{validation['unchanged_by_structure']}窗由原图逐节点/逐权重一致及非0门控证明输出保留，没有冒称全部960窗直接调用新文件。",'',
        '原始输入核对使用相同候选图追加三个已有中间输出作观测，没有改任何计算节点；最终单输出文件另通过边界、实际触发与不触发样本的数值/资源检测。全部相关原始文件哈希与历史实际模型审计一致。',
        '',f"CPU4线程最慢样例均值{max(r['cpu_mean_s'] for r in check['cases']):.4f}秒，文件{check['bytes']:,}字节；接口、嵌入权重、内存、<100MB与<0.7秒本地检查通过。",'',
        '触发时返回原始视角的去偏置分数，其他输入完整保留0.48输出；不删除原有偏置常量、不改原权重、没有新增一次主干推理。0.48预测为1/2/3/4的结果全部结构性不变。',
        '','本地纠正主要仍在g1，空房代价集中于a1/e1。这种分布偏差正是平台可能涨也可能跌的原因，不能把本地净增12窗换算成平台提升。该候选供下一次平台反馈验证，0.48继续保留。',
        '','完整权重、分组、逐记录错误、源文件哈希与验证位于 `D:/huawei2026-experiments/pair_evidence_20261024/`。']
    (ROOT/'docs/PAIR_EVIDENCE_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (destination/'README.md').write_text('未评测探索候选：01_temporal_consensus80.onnx。平台最佳仍0.48。详见docs/PAIR_EVIDENCE_20261024.md。\n',encoding='utf-8')
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=digest,validation=validation),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
