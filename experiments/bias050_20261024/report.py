"""Publish the planned .50-bias experiment only after all checks pass."""
import csv
import shutil
from pathlib import Path
from build import ROOT,WORK,BEST,BEST_HASH,shared

EXPORT=WORK/'export'


def main():
    destination=ROOT/'models/bias050_20261024';target=destination/'01_final_bias_one050.onnx'
    if target.with_suffix('.json').exists() and shared.read(target.with_suffix('.json')).get('platform_score') is not None:
        print('Scored artifact exists; preserve later feedback.');return
    check=shared.read(EXPORT/'resource_check.json');validation=shared.read(EXPORT/'validation.json')
    comparison=shared.read(EXPORT/'comparison049.json');digest=shared.sha(EXPORT/'model.onnx')
    assert check['status']=='pass' and check['sha256']==validation['candidate_sha256']==comparison['candidate_sha256']==digest
    assert shared.sha(BEST)==BEST_HASH and comparison['only_changed_initializer']=='__final_bias_delta'
    raw=shared.read(EXPORT/'raw_checks.json');raw_keys={(r['identifier'],r['start']) for r in raw}
    changes=shared.read(EXPORT/'changes_vs049.json')
    assert all((r['identifier'],r['start']) in raw_keys for r in changes)
    destination.mkdir(parents=True,exist_ok=True)
    if target.exists():assert shared.sha(target)==digest
    else:shutil.copy2(EXPORT/'model.onnx',target)
    metadata=dict(id='final_bias_one050_20261024',name='最终1人总偏置0.50',path=target.relative_to(ROOT).as_posix(),
        bytes=target.stat().st_size,sha256=digest,platform_score=None,status='exploratory_awaiting_platform',
        report='docs/BIAS050_20261024.md',scored_control_sha256=BEST_HASH,scored_control_accuracy=.49,
        original048_sha256=shared.SOURCE_HASH,total1_bias=.5,extra_vs049=.15,
        method='Exact scored049 computation graph; replace only final residual constant0.35 with0.50. No extra Add, rerouting or backbone training.',
        goal='Explore whether platform can exceed0.52; not a prediction',
        limitation='Local training-source labels, development choice. Additional false positives expected; platform score unknown.',
        local_validation=validation,comparison049=comparison,resources=check)
    shared.save(target.with_suffix('.json'),metadata)
    shared.save(ROOT/'checks/bias050_20261024.json',metadata)
    registry=shared.read(ROOT/'models/registry.json')
    assert registry['current_best']['sha256']==BEST_HASH and registry['current_best']['platform_score']==.49
    registry['research_candidates']=[v for v in registry.get('research_candidates',[]) if v.get('id')!=metadata['id']]+[metadata]
    registry['next_submission']={k:metadata[k] for k in ('id','path','sha256','platform_score','status','report')}
    registry['research_status']='bias050_awaiting_platform'
    registry['latest_research']=dict(report=metadata['report'],status='planned_bias050_candidate_checked',best_scored=.49)
    registry['next_stage_goal']='exceed0.52 (unmet)'
    shared.save(ROOT/'models/registry.json',registry)
    csv_path=ROOT/'models/platform_scores.csv'
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not any(v['sha256']==digest for v in rows):
        rows.append(dict(model='原048最终1人总偏置0.50',platform_score='',path=metadata['path'],sha256=digest,
            source='planned followup after049; platform not evaluated',status='exploratory_awaiting_platform',submission_id='',
            availability='local',original_path=metadata['path'],archive_manifest=''))
        with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# 获分0.49后的总偏置0.50实验','',
        f"下一份提交文件：`{metadata['path']}`，{metadata['bytes']:,}字节，SHA256 `{digest}`。平台未评测；当前最佳0.49原件保持不变。",'',
        '按用户批准的NEXT_ROUTE_AFTER049路线，只测试总1人偏置0.50。目标是探索突破0.52，目标不等于预测。没有新训练、数据扩充、重选校准方式或结构叠加。',
        '','## 与0.49的精确差别','',
        '当前0.49为原始0.48最终输出加[0,0.35,0,0,0]。这次改为[0,0.50,0,0,0]，相对当前多0.15；不是在当前文件上再加0.50变成0.85。',
        '','对两份实际ONNX逐项比较：所有计算节点、学习权重、内部偏置、路由与接口均相同；唯一变化的initializer为__final_bias_delta，其中1个float32由0.35变0.50。元数据另外注明未评测与实验来源，没有新增第二个Add或主干推理。',
        '','只抬高1人分数；0/2/3/4之间的相对分数不变，但其中一类可能被1人超过。不能称为概率增加15个百分点。',
        '','## 本地诊断','',
        '| 模型 | 平台 | 本地正确/960 | 相对当前0.49纠正/改坏 |','|---|---:|---:|---:|',
        '| 总偏置0.35 | 0.49 | 688 | — |',
        '| 总偏置0.50 | 未测 | 692 | 8/4 |','',
        '源数据相对当前只改判12窗，净增4。相对原0.48纠正51/改坏13。追加收益来自1人，追加损失为4个空房误报；本地2/3/4正确数仍为66/78/140。小幅改变更利于单变量归因，但不意味着平台至少涨多少。',
        '','## 数值与资源核对','',
        f"960窗由此前实际0.48输出加float32残差推导，不冒称候选直接推理960次。另直接核对{validation['directly_raw_checked']}个原始输入，包括相对原0.48的全部{validation['all_changed_windows_directly_checked']}个改判窗口、相对当前0.49的全部12个改判窗口及边界样例；原缓存差与参考差均为0。原始数据SHA与历史审计一致。",'',
        f"单输出ONNX与独立原0.48+NumPy参考比较通过。CPU4线程最慢样例均值{max(v['cpu_mean_s'] for v in check['cases']):.4f}秒，峰值提交内存{check['peak_commit_bytes']/1e9:.3f}GB，文件{metadata['bytes']/1e6:.2f}MB，接口、嵌入权重、<0.7秒/<6GB/<100MB检查通过。",'',
        '所有本地样本参与过底座训练，幅度也在开发缓存上检查过，不是独立验证。上一轮高人数模型相似本地分数却平台下降，故不把72.083%换算成平台预测。',
        '','若平台提高，再按路线考虑一个相邻幅度；若持平或下降，保留0.49并转向边界范围消融，不直接追到更大偏置。',
        '','完整证明、12窗列表和原始核对：`D:/huawei2026-experiments/bias050_20261024/export/`。']
    (ROOT/'docs/BIAS050_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (destination/'README.md').write_text('未评测下一份模型01_final_bias_one050.onnx：最终1人总偏置0.50，比获分0.49版多0.15。目标探索突破0.52，平台尚未验证。详见docs/BIAS050_20261024.md。\n',encoding='utf-8')
    shared.save(WORK/'published.json',metadata)
    print(dict(path=str(target),bytes=metadata['bytes'],sha256=digest,platform_score=None,source_correct=692,
        source_corrected_vs049=8,source_broken_vs049=4,cpu_max_mean_s=max(v['cpu_mean_s'] for v in check['cases'])))


if __name__=='__main__':main()
