"""Verify the actual-artifact audit, plot confusion, and write an error report."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('D:/huawei2026-experiments/actual047_audit_20261024')
NAMES = ('base044', 'light047', 'full047')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert read(WORK / 'complete.json')['status'] == 'completed'
    summary = read(WORK / 'summary.json')
    with (WORK / 'all_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 960
    labels = np.array([int(r['truth']) for r in rows])
    predictions = {name: np.array([int(r[name + '_prediction']) for r in rows]) for name in NAMES}
    verify = {}
    for name in NAMES:
        matrix = np.bincount(labels * 5 + predictions[name], minlength=25).reshape(5, 5)
        assert matrix.tolist() == summary['models'][name]['confusion_matrix']
        verify[name] = dict(csv_confusion_matches=True)
    protected = np.isin(predictions['light047'], [0, 2])
    protected_disagreements = int((protected & (predictions['light047'] != predictions['full047'])).sum())
    # A numerical guard disagreement must be reported rather than silently ignored.
    summary['guard_diagnostic'] = dict(light_final_0_or_2_windows=int(protected.sum()),
        light_final_0_or_2_errors=int((protected & (predictions['light047'] != labels)).sum()),
        full_disagrees_on_guarded_labels=protected_disagreements,
        note='Full protects the final light decision. This does not identify every inner guard activation of light.')
    errors = {}
    for name in NAMES:
        pairs = Counter((int(y), int(p)) for y, p in zip(labels, predictions[name]) if y != p)
        errors[name] = [dict(truth=a, prediction=b, windows=count) for (a, b), count in pairs.most_common(8)]
    summary['largest_error_pairs'] = errors
    registry = read(ROOT / 'models/registry.json')
    fixed_bias = np.array(next(m['bias'] for m in registry['retained_models'] if m['id'] == 'best_044'))
    base_log = np.array([[float(r[f'base044_logit_{k}']) for k in range(5)] for r in rows])
    unbiased_prediction = (base_log - fixed_bias).argmax(-1)
    unbiased_cm = np.bincount(labels * 5 + unbiased_prediction, minlength=25).reshape(5, 5)
    summary['bias_diagnostic'] = dict(method='Subtract the exact final044 bias from actual044 outputs only; not a new047 policy or exact original037 artifact.',
        fixed044_bias=fixed_bias.tolist(), in_sample_unbiased_correct=int((unbiased_prediction == labels).sum()),
        confusion_matrix=unbiased_cm.tolist(), recall=(unbiased_cm.diagonal() / unbiased_cm.sum(1)).tolist(),
        limitations='This diagnoses local output shifting. Removing a bias that improved platform scores is not recommended from training-source labels alone.')
    # One fixed exploratory rule, not a threshold sweep or hidden-set claim.
    base_probability = np.exp(base_log - base_log.max(-1, keepdims=True))
    base_probability /= base_probability.sum(-1, keepdims=True)
    original_one = (predictions['base044'] == 1) & (base_probability[:, 1] >= .8)
    rescued = predictions['light047'].copy(); rescued[original_one] = 1
    summary['exploratory_original_one_guard'] = dict(
        rule='If actual044 original-view softmax class1 >=.8, preserve its class1; otherwise use actual light047. One fixed diagnostic, no sweep.',
        triggered_windows=int(original_one.sum()),
        corrected=int(((predictions['light047'] != labels) & (rescued == labels)).sum()),
        broken=int(((predictions['light047'] == labels) & (rescued != labels)).sum()),
        in_sample_correct=int((rescued == labels).sum()),
        note='Post-hoc exploratory analysis of training-source errors; threshold is not a calibrated correctness guarantee. No candidate exported.')
    argmax_guarded = predictions['light047'].copy()
    argmax_guarded[predictions['base044'] == 1] = 1
    summary['exploratory_original_one_argmax_guard'] = dict(
        rule='Preserve actual044 original-view argmax1, no score threshold; otherwise use light047.',
        triggered_windows=int((predictions['base044'] == 1).sum()),
        corrected=int(((predictions['light047'] != labels) & (argmax_guarded == labels)).sum()),
        broken=int(((predictions['light047'] == labels) & (argmax_guarded != labels)).sum()),
        in_sample_correct=int((argmax_guarded == labels).sum()),
        note='Second fixed exploratory rule, post-hoc training-source diagnostic only. It can override the old0/2 guard; no threshold tuning, export or hidden-score inference.')
    # Fixed blend is diagnostic only: these labels are in-sample, not a selection set.
    probability = {}
    for name in ('light047', 'full047'):
        z = np.array([[float(r[f'{name}_logit_{k}']) for k in range(5)] for r in rows])
        p = np.exp(z - z.max(-1, keepdims=True)); probability[name] = p / p.sum(-1, keepdims=True)
    equal_blend = (probability['light047'] + probability['full047']).argmax(-1)
    union = (predictions['light047'] == labels) | (predictions['full047'] == labels)
    summary['complementarity_diagnostic'] = dict(
        in_sample_fixed_equal_probability_blend_correct=int((equal_blend == labels).sum()),
        in_sample_label_oracle_union_correct=int(union.sum()),
        note='Oracle uses true labels and is not an implementable router. Equal blend here is diagnostic, not a new recommended submission.')
    for relative, digest in [
        ('models/best_047_light/model.onnx', summary['model_hashes']['light047']),
        ('models/best_047_full/model.onnx', summary['model_hashes']['full047'])]:
        with (ROOT / relative).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
    save(WORK / 'summary.json', summary)
    save(ROOT / 'checks/actual047_in_sample_audit.json', summary)
    save(WORK / 'report_audit.json', dict(status='pass', source_rows=960,
        metrics_verified=verify, original_best_hashes_unchanged=True,
        protected_label_disagreements=protected_disagreements))

    figures = ROOT / 'docs/figures'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for ax, name, title in zip(axes, ('light047', 'full047'), ('4-view, submission49130', '12-view, submission49131')):
        cm = np.array(summary['models'][name]['confusion_matrix'])
        recall = cm / cm.sum(1, keepdims=True)
        image = ax.imshow(recall, vmin=0, vmax=1, cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{cm[i,j]}\n{100*recall[i,j]:.1f}%', ha='center', va='center',
                        fontsize=9, color='white' if recall[i,j] > .55 else '#172b4d')
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xlabel('Predicted people'); ax.set_ylabel('Actual people'); ax.set_title(title)
    fig.colorbar(image, ax=axes, label='Recall within actual class', shrink=.8)
    fig.suptitle('Actual scored models on labelled training-source data\nIn-sample diagnostic, NOT hidden-test accuracy', fontsize=12)
    fig.savefig(figures / 'actual047_confusion.png', dpi=170)
    plt.close(fig)

    light, full = [summary['models'][name] for name in ('light047', 'full047')]
    class_lines = '\n'.join(f"| {i}人 | {light['support'][i]} | {100*light['recall'][i]:.2f}% | {100*full['recall'][i]:.2f}% |"
        for i in range(5))
    room_lines = '\n'.join(f"| {room} | {values['light047']['windows']} | {100*values['light047']['in_sample_accuracy']:.2f}% | {100*values['full047']['in_sample_accuracy']:.2f}% |"
        for room, values in summary['by_room'].items())
    record_lines = '\n'.join(f"| {r['identifier']} | {r['headcount']} | {r['light047_correct']}/20 | {r['full047_correct']}/20 |"
        for r in summary['worst_records'][:8])
    top_pairs = '\n'.join(f"- {name}: " + '；'.join(f"真实{x['truth']}人→预测{x['prediction']}人：{x['windows']}窗" for x in errors[name][:5])
        for name in ('light047', 'full047'))
    delta = summary['transitions']['light047_to_full047']
    base_delta = summary['transitions']['base044_to_light047']
    recommendations = [
        '先检查下表最差原始记录是否有异常采集、时间间隔或标签来源问题。模型预测不能证明标签错，不能自动改标签。',
        '把真实0.47在本地的错误类型与留房间代理分开：若原权重已能识别本地3/4人，而代理失败，后续重点应是跨房间保留该表示，而非声称实际0.47没有学会3/4人。',
        '区分原0.44已错、多视角纠正、以及增强改错三组记录。对共同错误优先检查信号表示/校准；对增强改错优先检查视角可靠性。训练数据上的哪个方案更准，不足以决定隐藏平台选哪个。',
        '若只少量记录贡献大量错误，先做记录级稳健性和采集差异诊断，不通过重叠窗口把同一条错误记录重复成大量独立证据。',
        '所有后续训练/校准都保留记录与房间隔离；不得用实际全数据0.47作为外层留房间教师后把该验证称为独立。']
    bias_lines = '\n'.join(f"| {i}人 | {100*summary['models']['base044']['recall'][i]:.2f}% | {100*summary['bias_diagnostic']['recall'][i]:.2f}% |"
        for i in range(5))
    original_one_audit = summary['exploratory_original_one_guard']
    original_one_argmax = summary['exploratory_original_one_argmax_guard']
    report = f'''# 实际0.47模型：本地错误审计

## 数据与性质

运行的是原始提交文件49130（四视角）和49131（十二视角），并加入实际0.44文件对照，没有用重新训练代理替代。四视角及0.44逐窗使用未经修改的ONNX文件进行CPU推理。十二视角的原图在小版最终预测0/2时直接返回小版输出，因此对间隔大于0.005的这类窗口，在每条记录先用原文件核对该分支后复用数学上相同的输出，其余窗口直接跑原大版文件。大版实际ONNX调用{summary['full_execution_counts']['original_onnx']}窗，保护分支复用{summary['full_execution_counts']['proven_protected_branch_reuse']}窗；逐窗执行方式有单独字段，不把复用计为真实大版调用。原模型哈希核对通过，没有训练、拟合新偏置或修改模型。

覆盖全部48条有标签记录，每条12000帧切为20个不重叠600帧窗口，共960窗。每窗ATI最后置0，与既有提交接口一致。原始CSI和ATI文件哈希已保存，输入检查通过。这些原记录参与过基础权重训练，**全部结果是训练来源数据上的诊断，不能当作验证或老师隐藏测试准确率。** 每条记录内窗口相关，不是960个独立场景。

## 总体与逐人数

- 原0.44：{summary['models']['base044']['correct']}/960，{100*summary['models']['base044']['in_sample_accuracy']:.2f}%。
- 四视角0.47：{light['correct']}/960，{100*light['in_sample_accuracy']:.2f}%。
- 十二视角0.47：{full['correct']}/960，{100*full['in_sample_accuracy']:.2f}%。

这里的本地数字不改变平台登记的0.44/0.47/0.47。

| 真值 | 窗口数 | 四视角召回 | 十二视角召回 |
|---|---:|---:|---:|
{class_lines}

![实际提交模型混淆矩阵](figures/actual047_confusion.png)

主要混淆：

{top_pairs}

## 逐房间与记录

| 房间 | 窗口数 | 四视角本地正确率 | 十二视角本地正确率 |
|---|---:|---:|---:|
{room_lines}

房间人数分布不同，不能仅凭房间总体准确率评判房间难度。unknown保留原元数据，不猜测其真实房间。按四视角错误数优先排序的记录如下，完整48条见record_summary.csv：

| 原记录 | 真实人数 | 四视角正确窗 | 十二视角正确窗 |
|---|---:|---:|---:|
{record_lines}

## 增强的收益、损失与互补

原0.44→四视角：纠正{base_delta['corrected']}窗，改错{base_delta['broken']}窗。四视角→十二视角：纠正{delta['corrected']}窗，改错{delta['broken']}窗，共{delta['different_predictions']}窗预测不同，两者共同错误{delta['both_wrong']}窗。

四视角最终预测0/2的窗口为{int(protected.sum())}个，其中错误{int((protected & (predictions['light047'] != labels)).sum())}个；十二视角在这些受外层保护窗口上出现{protected_disagreements}次类别不同。该统计不能直接代表小版内层保护是否触发。

固定等权概率融合的本地正确数为{summary['complementarity_diagnostic']['in_sample_fixed_equal_probability_blend_correct']}/960。若借助真实标签总挑两版中正确的一版，理论并集为{int(union.sum())}/960；这个并集需要答案，不是可部署模型。这里只诊断互补性，没有用这些训练数据选择新提交。

## 固定偏置的本地作用

仅从实际0.44输出减去其最后五维固定偏置，得到去偏置诊断（不是重新运行原0.37文件，也没有重算0.47的保护门）。该诊断本地正确{summary['bias_diagnostic']['in_sample_unbiased_correct']}/960。

| 真值 | 实际0.44召回 | 0.44末尾去偏置诊断召回 |
|---|---:|---:|
{bias_lines}

这能帮助区分基础信号表示与最终分数偏移，但不能据此推荐删除偏置：偏置已在平台产生过收益，训练来源分布和隐藏分布不同。本地校准更好不等于隐藏更好。

## 一个窄范围保护的探索性诊断

观察到原0.44正确而增强改错后，固定检查一个规则：仅当原视角0.44的1人softmax分数至少0.8时保留其1人判断，其他全部沿用四视角0.47。不扫阈值。它触发{original_one_audit['triggered_windows']}窗，纠正{original_one_audit['corrected']}窗、改错{original_one_audit['broken']}窗，本地正确{original_one_audit['in_sample_correct']}/960。

这是看过训练来源错误后的探索，**不是独立验证，也没有导出提交文件**。0.8分数不等于80%真实可信度。若后续测试该方向，应把原0.47保持为对照，并明确它可能覆盖原0/2保护、可能在隐藏集引入新的错误，不能把本地收益换算为平台涨幅。

另一个固定消融是不设分数阈值、只保留原视角argmax为1的判断：触发{original_one_argmax['triggered_windows']}窗，纠正{original_one_argmax['corrected']}窗、改错{original_one_argmax['broken']}窗，本地正确{original_one_argmax['in_sample_correct']}/960。它同样是事后探索，并可能覆盖原0/2保护，不是已验证的升级。没有继续扫描阈值选最优点。

## 如何据此改进

四视角1人错误为342窗，占其441个错误的77.55%。这使下一轮优先级改变：先检验原始/变换视角冲突的处理，再扩大结构。原视角0.44在本地预测1人的精确率为155/158=98.10%，但召回仅43.06%，所以单独保留这些判断也救不回全部1人错误。

建议先做两个独立实验：

1. **冻结权重的原视角保护消融。** 把“原视角1人判断是否保留”作为一个小改动，与原0.47对照；这是探索候选，不能凭当前训练数据直接宣布平台会提升。0.8阈值没有触发样本，说明不能凭直觉设置信心阈值。
2. **带真实标签的变换一致性微调。** 从已有权重出发，训练时同时看原输入、时间反转及少量接收变换；保留原视角人数损失和表示约束，再对变换后的相同人数监督。分别比较只微调读出、残差适配器、适配器加一致性，避免继续只在推理时使用模型未充分适应的视角。标签监督优先，不能强迫学生复制教师在1人样本上的错误。

在这两个实验稳定后，再考虑0/1精细区分与2/3/4细分的结构变化。不能将新分支直接命名为静止、慢走、快走检测器，因为没有这些标签。先用实际模型错误指导消融，而非以代理3/4人失败为由扩大网络。

{chr(10).join('- ' + r for r in recommendations)}

本报告定位的是已知数据上的实际错误。老师隐藏集哪些类错、各类比例及具体房间仍不可见，不能把这些错误比例直接当作隐藏集分布，也不能承诺修正这些记录就达到0.54。

## 文件

实验目录：`{WORK.as_posix()}`。

- errors.csv：任一0.47预测错误的窗口，含原记录、帧范围、真实人数、两版预测及分数。
- all_predictions.csv：960窗完整输出，含0.44对照。
- record_summary.csv：48条记录的正确数及预测分布。
- summary.json：总体、逐类、逐房间、纠正/改错及哈希。
- records/：逐原记录检查点，可从中断处继续。

errors.csv中的softmax最大值只是输出分数，不是经独立验证的正确概率。relative_frame_change只是原始幅度相邻变化指标，不是真实运动速度或静止/慢走/快走标签。
'''
    (ROOT / 'docs/ACTUAL047_ERROR_AUDIT.md').write_text(report, encoding='utf-8')
    registry_path = ROOT / 'models/registry.json'
    registry = read(registry_path)
    registry['latest_actual_model_audit'] = dict(report='docs/ACTUAL047_ERROR_AUDIT.md',
        artifact_root=str(WORK), windows=960, records=48, independent_test=False, parameters_changed=False)
    save(registry_path, registry)
    print(json.dumps(dict(report='docs/ACTUAL047_ERROR_AUDIT.md', models=summary['models'],
        transitions=summary['transitions'], guard=summary['guard_diagnostic']), ensure_ascii=False))


if __name__ == '__main__':
    main()
