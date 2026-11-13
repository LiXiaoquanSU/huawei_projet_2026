"""Publish a concise description after validation and resource checks pass."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/original_one_guard_20261024')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    meta = read(WORK / 'candidate.json')
    path = ROOT / meta['path']
    if path.with_suffix('.json').exists():
        meta = read(path.with_suffix('.json'))  # Later user feedback takes precedence.
    resource = meta['resource']; validation = meta['validation']
    assert sha(path) == meta['sha256'] == resource['sha256']
    assert validation['status'] == resource['status'] == 'pass'
    platform_note = ('平台成绩未知，目标0.54，不是已经达到0.54。原两份0.47仍是已确认最佳。'
        if meta['platform_score'] is None else
        f"用户报告平台成绩{meta['platform_score']:.2f}，当前已确认最佳；较49130的0.47显示提高1个百分点，目标0.54尚未达到。提交编号未提供。")
    worst_time = max(c['cpu_mean_s'] for c in resource['cases'])
    table = '\n'.join(f'| {i}人 | {old:.2f}% | {100 * new:.2f}% |' for i, (old, new) in
        enumerate(zip([98.6363636364, 5., 55., 65., 100.], validation['recall'])))
    report = f'''# 保留原视角1人判断

文件：`{meta['path']}`，**{meta['bytes']/1e6:.2f} MB**。

SHA256：`{meta['sha256']}`。

**{platform_note}** 这次没有重新训练、没有改偏置、没有使用外部数据或U数据。

## 准确规则

仍计算原0.47四个视角。取其中未经时间/天线变换的原0.44分支输出：

- 若该原视角argmax为1，输出该原视角的五维分数。
- 否则输出原四视角0.47的五维分数。

这个判断只看输入数据产生的模型输出，不读取真实人数、房间名称、记录编号或其他窗口。没有置信度阈值，没有按验证集搜索偏置。**原视角判1时，新外层规则可能覆盖旧0/2保护。** 其他样本保持原0.47流程。

## 具体改动

原文件为49130，SHA256 `d41c3d7cff9a31cde80635ea40df30b4f432c9029b50b40430fff7af00ee62ad`。原9061个节点全部保留（仅重命名最终内部输出），原185个initializer逐字节不变，包括全部学习权重、固定偏置和旧变换/保护常量。只新增ArgMax、Equal、Where三个节点及一个int64常量1。

新增规则不再增加神经网络前向次数，但也不把Where当成提前退出：四个视角仍然计算。单文件嵌入权重，接口与原49130相同。

## 实际导出文件的验证

重新加载原始CSI/ATI，直接运行这份导出的ONNX，覆盖48条记录、960个不重叠窗口。与独立保存的实际0.44和49130输出按上述规则计算的参考逐窗比较：

- {validation['argmax_matches_expected_policy']}/960最终类别符合规则。
- 触发新规则{validation['triggered']}窗，另外{validation['unchanged_route_checks']}窗保持旧规则分支。
- 最大概率数值误差{validation['max_probability_error']:.10g}，最大log分数误差{validation['max_logit_error']:.10g}。
- 本地正确{validation['correct']}/960，即68.125%（约68.13%）；原四视角519/960，即54.06%。纠正{validation['corrected']}窗，改错{validation['broken']}窗。

| 真实人数 | 原四视角召回 | 新候选召回 |
|---|---:|---:|
{table}

两处新增错误均为原本判对的空房改成1人，不能只报告净收益而省略代价。这些源记录参与过基础权重训练，规则也是看过其错误后提出的，因此**68.13%不是独立验证，更不能换算成平台涨幅**。这里只证明导出的文件实现了预期规则及其本地作用。

## 运行与文件约束

隔离CPU进程执行普通输入、异常时间间隔及全零输入，每例20次计时；接口、有限数值、参考输出、体积和内存检查全部通过。

- 文件{meta['bytes']:,}字节，低于100,000,000字节。
- CPU四线程三例中最慢均值{worst_time:.4f}秒，低于0.7秒。
- 峰值提交内存{resource['peak_commit_bytes']/1e9:.3f}GB，峰值工作集{resource['peak_working_set_bytes']/1e9:.3f}GB。

这是当前机器的局部运行检查，不保证所有机器有相同速度。原0.44/0.47哈希未变。没有自动上传比赛平台；平台成绩由用户后续反馈登记，不改动ONNX本体。

## 复现

源码：experiments/original_one_guard_20261024/build.py。运行输出及逐记录检查点位于 `{WORK.as_posix()}`；再次运行会复用同哈希候选的验证记录。需要本机D盘数据及上一轮实际模型审计缓存。

资料包括structure_proof.json、validation.json、validation_records/、resource_check.json、candidate.json、references.json和source_snapshot/。本轮只交付这一份候选，避免同时提交多份不清晰的变化。
'''
    (ROOT / 'docs/ORIGINAL_ONE_GUARD_20261024.md').write_text(report, encoding='utf-8')
    score_line = '平台成绩尚未取得。' if meta['platform_score'] is None else f"用户报告平台成绩{meta['platform_score']:.2f}，当前最佳。"
    (path.parent / 'README.md').write_text(
        '# 原视角1人保护版\n\n文件为本目录的 `01_original_one_guard.onnx`。' + score_line + '\n\n'
        '保留原0.44视角的1人判断，其余沿用49130四视角0.47。没有重新训练或改偏置。\n\n'
        '本地训练来源诊断68.13%，不代表平台成绩；详细验证见 docs/ORIGINAL_ONE_GUARD_20261024.md。\n', encoding='utf-8')
    registry_path = ROOT / 'models/registry.json'
    registry = read(registry_path)
    registry['target_platform_score'] = .54
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if meta['platform_score'] is not None:
        (WORK / 'candidate.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    snapshot = WORK / 'source_snapshot'; manifest = []
    for source in (Path(__file__), Path(__file__).with_name('build.py'), ROOT / 'tools/check_candidate.py'):
        relative = source.relative_to(ROOT); target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        manifest.append(dict(path=relative.as_posix(), sha256=sha(source)))
    (WORK / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(path), size_MB=meta['bytes']/1e6,
        cpu_worst_mean_s=worst_time, platform_score=meta['platform_score'], report='docs/ORIGINAL_ONE_GUARD_20261024.md')))


if __name__ == '__main__':
    main()
