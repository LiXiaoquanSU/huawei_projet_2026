"""Consolidate completed stages without promoting an unvalidated model."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/environment_protocol_20261025')
PACKAGE = ROOT / '2026-10-25_037配方与种子复现'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    graph = read(WORK / 'graph/results.json')
    selected = read(WORK / 'graph/selection.json')['selected']
    fits = sorted((WORK / 'graph').glob('*_s*_*.pt'))
    assert len(fits) == 24
    audit = []
    grouped = {}
    for path in fits:
        d = torch.load(path, weights_only=True)
        assert set(d['target_adaptation_records']).isdisjoint(d['target_test_records'])
        assert set(d['source_fit_records']).isdisjoint(d['target_adaptation_records'] + d['target_test_records'])
        assert d['final_invariance_error'] < 1e-5
        grouped.setdefault((d['room'], d['seed']), set()).add(d['labelled_draws_sha256'])
        audit.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          parameters=d['parameters'], step=d['step'], amount=d['amount'],
                          invariance_error=d['final_invariance_error']))
    assert all(len(values) == 1 for values in grouped.values())
    (WORK / 'graph/fit_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    report_path = ROOT / 'docs/ENVIRONMENT_PROTOCOL_20261025.md'
    report = report_path.read_text(encoding='utf-8').split('\n## 第二阶段：')[0]
    report = report.replace('# 环境适应研究：两套验证与48次小模块训练', '# 双验证研究：72次训练与环境子空间对照')
    lines = ['', '## 第二阶段：条件环境子空间对照', '',
             '从源数据中提取“同人数、不同房间”的均值残差，并把这些方向投影到人数类别均值空间的正交补。该空间只是候选环境方向，不能认定已经分离出纯房间因素。源校准只选择0、0.25、0.5强度；目标标签不参与选择。', '',
             '|维数|无目标U净增/880|目标U背景参考净增/880|U相对无U净增|',
             '|---|---:|---:|---:|']
    sub = read(WORK / 'subspace_results.json')
    for rank in [2, 4]:
        source = next(x for x in sub if x['rank'] == rank and x['kind'] == 'projection')
        target = next(x for x in sub if x['rank'] == rank and x['kind'] == 'reference')
        lines.append(f"|{rank}|{source['net']:+d}|{target['net']:+d}|{target['matched_source_net']:+d}|")
    lines += ['', '最好的目标参考方案仅净增3窗，未达到预设2个百分点候选门槛。', '',
              '## 第三阶段：输入天线关系图', '',
              '- 三个接收通道共享幅度编码器；六条有向边编码天线对相位余弦、正弦和两端动态幅度，反向边正确处理相位共轭。两层消息传递后对天线做对称汇聚，再加入时间间隔和时间注意力。',
              '- 原054主干和原视角1人/双时间0或2人保护分支保留。新分支仅参与完整12视角fallback；不要求一个置换不变学生拟合原视角优先的最终规则。',
              f"- 新分支{audit[0]['parameters']:,}参数。四房间×两种子×源训练/两折目标U共24次训练；每次150步，源校准在50/150步和0/25%/50%融合中选择。目标U使用软教师蒸馏，不使用真实标签或每房间均匀假设。",
              '- 24份权重核验：同房间同种子有标签抽样SHA相同；适应/测试记录互斥；六种Rx编号变换的预测误差均小于1e-5。', '',
              '|协议|正确/880|净增窗口|a1变化|c1变化|g1变化|e1变化|按记录95%提升区间|相对源图模型净增|',
              '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in graph:
        gains = [f"{100*r['room_gains'][room]:+.2f}" for room in ['a1', 'c1', 'g1', 'e1']]
        lo, hi = r['ci95']
        correct = sum(x['correct'] for x in r['records'])
        lines.append('|'+ '|'.join([r['protocol'], f'{correct}/880', f"{r['net']:+d}", *gains,
                                      f'{100*lo:+.2f}～{100*hi:+.2f}', f"{r['matched_source_net']:+d}"] )+'|')
    lines += ['', '图分支错题变化：']
    for r in graph:
        rescued = sum(x['metrics']['corrected'] for x in r['rooms'].values())
        broken = sum(x['metrics']['broken'] for x in r['rooms'].values())
        lines.append(f"- {r['protocol']}：救回{rescued}窗、改坏{broken}窗。")
    guards = []
    for room in ['a1', 'c1', 'g1', 'e1']:
        evidence = torch.load(Path('D:/huawei2026-experiments/subnet_expectation_20261025/evidence') / f'{room}.pt', weights_only=True)
        calibration = read(Path('D:/huawei2026-experiments/protected_context_20261024/results') / f's42_{room}/metrics.json')['calibration']['baseline']
        raw = evidence['raw']; bias = torch.tensor(calibration['bias']); temperature = calibration['temperature']
        v0 = raw[:,0,0]/temperature + bias
        v45 = raw[:,0].exp().mean(1).clamp_min(1e-12).log()/temperature + bias
        protected = (v0.argmax(-1)==1) | (v45.argmax(-1)==0) | (v45.argmax(-1)==2)
        ev = torch.load(WORK / f'graph/eval_DG_{room}.pt', weights_only=True)
        ix = ev['indices']; wrong = ev['baseline'].argmax(-1) != ev['labels']
        guards.append(dict(room=room, total=len(ix), protected=int(protected[ix].sum()),
                           errors=int(wrong[ix].sum()), protected_errors=int((protected & wrong)[ix].sum()),
                           editable_errors=int((~protected & wrong)[ix].sum())))
    (WORK / 'graph/guard_diagnostic.json').write_text(json.dumps(guards, indent=2), encoding='utf-8')
    allerrors = sum(x['errors'] for x in guards); locked = sum(x['protected_errors'] for x in guards)
    lines += ['', f'保护规则的可修改空间诊断：对照共错{allerrors}窗，其中{locked}窗落在固定保护分支，fallback分支能接触的错误仅{allerrors-locked}窗。这些固定分支内的错误无法通过本轮图分支纠正。该结果限于历史留房间代理，不能外推老师测试集比例。',
              '这说明“新分支收益有限”同时受结构/训练和固定保护覆盖面影响，不能只归因于图结构不合适。下一轮若研究有条件解除保护，必须单独检查被保护的正确样本损失；此前完全取消保护平台下降的经验仍有效。']
    lines += ['', '变化/区间单位为百分点；对照278/880为留房间历史教师，不是原054在全部训练数据上的成绩。区间是在固定四个开发房间内部按整记录重采样，不能代表新房间总体的不确定性。', '',
              '## 本轮总决策', '',
              f'共完成48次小模块训练、24次图分支训练及两种维数的子空间对照。图分支通过门槛的协议：{selected or "无"}。',
              '门槛为整体至少+2个百分点、至少两房间受益、任何房间退步不超过2个百分点、按记录提升区间下界大于0；目标U还需明确优于匹配的源训练图模型。',
              '研究用目标U来自留出房间的标签遮蔽记录，用来模拟目标环境可见；没有把官方150条U用于这72次训练。官方数据部分完成ATI全扫描与11条CSI抽查，不能称全量U已用于训练。',
              '关键发现：同一房间的适应组与测试组也可能类别支持不同。e1一折适应组全空房，测试组全1人；强行对齐整体类别分布不成立。',
              '已反复使用的四个开发房间不构成全新盲测。此轮结果只用于淘汰缺乏支持的方向；没有新增平台成绩。',
              '下一步应优先补足其他物理房间的2–4人监督，或先验证能否在不依赖人数分布的条件下估计跨记录背景；现有结果不支持继续放大同一种自训练损失。',
              '', '全部新权重、输出、计划、划分与审计在D盘上述目录保留；未删除历史最佳模型、未提交平台、未推送Git。']
    if selected is None:
        lines += ['', '本轮不导出新的推荐ONNX，平台最佳仍为已评分0.54，目标0.60尚未达到。']
    report += '\n'.join(lines) + '\n'
    report_path.write_text(report, encoding='utf-8')
    (PACKAGE / '环境适应双验证研究.md').write_text(report, encoding='utf-8')
    print(json.dumps(dict(fits=72, graph_selected=selected, graph_parameters=audit[0]['parameters'], report=str(report_path)), ensure_ascii=False))

if __name__ == '__main__':
    main()
