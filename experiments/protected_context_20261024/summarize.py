"""Audit completed artifacts and produce the local experiment report."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/protected_context_20261024')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    complete = read(WORK / 'complete.json')
    assert complete['status'] == 'completed'
    assessment = complete['assessment']
    rows = []; proof = []
    for folder in sorted((WORK / 'results').iterdir()):
        manifest = read(folder / 'manifest.json')
        groups = [{r['identifier'] for r in manifest[k]} for k in ('fit', 'calibration', 'test')]
        assert all(groups[a].isdisjoint(groups[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        assert len(groups[1]) == 5
        with np.load(folder / 'predictions.npz') as data:
            a = data['baseline_calibrated']; b = data['protected_context25']
            protected = np.isin(a.argmax(-1), [0, 2])
            assert np.array_equal(a[protected], b[protected])
            assert np.array_equal(a.argmax(-1), data['agreement_control'].argmax(-1))
            for method in ('baseline_calibrated', 'context_calibrated', 'protected_context25'):
                assert np.isfinite(data[method]).all()
                assert np.max(np.abs(data[method].sum(-1) - 1)) < 1e-5
            assert len(data['labels']) == len(manifest['test'])
        metric = read(folder / 'metrics.json')
        proof.append(dict(fold=folder.name, disjoint_records=True,
            guard_outputs_exactly_preserved=True, agreement_control_labels_unchanged=True))
        if metric['seed'] == 42:
            values = [100 * metric['methods'][method]['accuracy'] for method in
                      ('baseline_calibrated', 'context_calibrated', 'protected_context25')]
            rows.append('| ' + metric['room'] + ' | ' + ' | '.join(f'{x:.2f}%' for x in values) + ' |')
    dependencies = read(WORK / 'source_manifest.json')['files']
    for relative, digest in dependencies.items():
        source = (ROOT / relative).resolve()
        assert source.is_relative_to(ROOT) and sha(source) == digest
        target = WORK / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    for source in (Path(__file__), Path(__file__).with_name('README.md')):
        target = WORK / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    save(WORK / 'artifact_audit.json', dict(status='pass', folds=proof,
        source_manifest_matches=True, best_models_modified=False))
    methods = assessment['methods']
    base = methods['baseline_calibrated']; candidate = methods['protected_context25']
    translations = {'at least one class recall regressed': '至少一种人数的召回下降：本轮1人由25.71%降至24.64%。'}
    reasons = '\n'.join('- ' + translations.get(reason, reason) for reason in assessment['rejection_reasons']) or '- 第一种子通过；仍须核对第二种子结果。'
    report = f'''# 主模型与上下文模型：三方隔离的校准/保护融合对照

本轮完成{complete['new_training_fits']}次新的5轮训练，使用比赛数据，没有外部数据和U数据。当前平台最佳仍为0.47；以下全部是新权重的本地代理结果，不是原0.47模型的隐藏成绩。新模型未导出、未自动提交。

## 方法

每折留出整间房用于外层评估；其余已知房间中按固定哈希规则为每个人数选择一条完整记录作校准，共5条、100窗，剩余记录训练。训练、校准、外层评估三个集合的源记录互斥，unknown不参与。基准结构和上下文结构分别初始化、训练，并用各自校准记录拟合带正则的偏置和温度。校准目标按这5条记录均衡类别，这是一种开发目标，并非声称隐藏类别均匀。

主路使用原/反转时间及原/反转接收顺序共4视角，保留0.47式保护；补充路为单视角上下文事件结构。固定75%主路、25%补充路概率融合，并保留主路最终预测0/2时的全部输出。不搜索融合权重或按外层标签选择校准参数。

## 第一种子42

| 外层房间 | 各自校准后的主路 | 各自校准后的上下文路 | 保护融合 |
|---|---:|---:|---:|
{chr(10).join(rows)}

a1/c1/g1三个开发房间等权均值：主路{100 * base['room_mean']:.2f}%，融合{100 * candidate['room_mean']:.2f}%；汇总宏召回：主路{100 * base['macro_recall']:.2f}%，融合{100 * candidate['macro_recall']:.2f}%。e1另作压力检查，不计入该均值。

| 真值人数 | 主路召回 | 融合召回 |
|---|---:|---:|
{chr(10).join(f'| {i} | {100*a:.2f}% | {100*b:.2f}% |' for i,(a,b) in enumerate(zip(base['recall'],candidate['recall'])))}

事先规定的门槛判断：

{reasons}

门槛不仅看总体平均，还要求各类召回不降、单房间退步不超过2个百分点、没有整间房单类塌缩，以及e1保留准确率且能预测两种实际存在的类别。若未通过，不为了让候选合格临时改变门槛，也不继续筛选随机种子。

## 能说明什么

独立校准确实避免了无条件沿用队友偏置，但没有自动解决人数与房间混杂。需要同时看逐类表，不能把少数人数的收益解释为完整0–4人计数能力提高。两模型预测有差异，也不等于固定平均能利用这些差异；逐记录纠正/改错保存在每折metrics.json。

本轮主路本身是重新训练的代理，并非真实0.47权重。每折为了校准少用5条训练记录，与历史留房间代理的训练记录数量也不同，不能把历史指标和本轮指标直接解释成进步或退步。外层房间已经在过去研究中反复使用；训练数据每种已知布局只有一个物理房间，所以外层同时改变布局，与老师布局家族内跨物理房间任务仍有差异。

即使校准记录与训练/评估严格隔离，单个人数只有一条校准记录，仍不足以代表该类跨房间分布。校准对某个新权重有效，也不能直接复制给真实0.47或另一个全数据checkpoint。

## 保存与复现

所有新权重、反向预处理特征、划分、预测、校准参数、源代码快照、判定和审计保存在 `{WORK.as_posix()}`。入口为 experiments/protected_context_20261024/run.py；本报告由summarize.py根据完成记录生成。原0.47文件未修改，平台成绩登记不变。
'''
    (ROOT / 'docs/PROTECTED_CONTEXT_20261024.md').write_text(report, encoding='utf-8')
    path = ROOT / 'models/registry.json'
    registry = read(path)
    registry['research_status'] = 'protected_context_round_completed_no_candidate'
    registry['latest_research'] = dict(report='docs/PROTECTED_CONTEXT_20261024.md',
        artifact_root=str(WORK), new_training_fits=complete['new_training_fits'],
        candidate_exported=False, first_seed_passed=assessment['eligible_for_second_seed'])
    save(path, registry)
    print(json.dumps(dict(report='docs/PROTECTED_CONTEXT_20261024.md', audit='pass',
        fits=complete['new_training_fits'], rejection_reasons=assessment['rejection_reasons']), ensure_ascii=False))


if __name__ == '__main__':
    main()
