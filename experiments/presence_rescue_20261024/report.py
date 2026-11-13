"""Summarize predeclared room/record A/B screen; do not force an export."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/presence_rescue_20261024')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def aggregate(results,kind,seed,reference):
    selected=[r for r in results if r['kind']==kind and r['seed']==seed]
    assert len(selected)==4
    totals={key:sum(r[reference][key] for r in selected) for key in (
        'windows','baseline_correct','correct','corrected','broken','empty_count','one_count',
        'rescued_one','added_empty_false_positive','standalone_binary_correct','binary_count')}
    totals['empty_recall_drop']=totals['added_empty_false_positive']/totals['empty_count']
    totals['one_recall_gain']=totals['rescued_one']/totals['one_count']
    totals['rooms_with_rescue']=[r['room'] for r in selected if r[reference]['rescued_one']>0]
    totals['rooms_with_net_gain']=[r['room'] for r in selected if r[reference]['corrected']>r[reference]['broken']]
    records=[record for r in selected for record in r[reference]['records']]
    totals['standalone_one_recall']=sum(r['auxiliary_present'] for r in records if r['truth']==1)/totals['one_count']
    totals['standalone_empty_recall']=1-sum(r['auxiliary_present'] for r in records if r['truth']==0)/totals['empty_count']
    totals['by_room']={r['room']:r[reference] for r in selected}
    totals['passed']=totals['empty_recall_drop']<=.01 and len(totals['rooms_with_net_gain'])>=2 and totals['corrected']>totals['broken']
    return totals


def main():
    results=read(WORK/'results.json'); assert len(results)==24
    groups={kind:{str(seed):{reference:aggregate(results,kind,seed,reference)
        for reference in ('actual048','independent_backbone_proxy')} for seed in (42,2026)}
        for kind in ('A_actual_diagnostic','A_proxy','B_signal')}
    # A_actual is a full-data-encoder diagnostic, never sufficient for export.
    eligibility={kind:all(groups[kind][str(seed)]['independent_backbone_proxy']['passed'] for seed in (42,2026))
                 for kind in ('A_proxy','B_signal')}
    summary=dict(groups=groups,eligible=eligibility,head_decomposition=read(WORK/'head_decomposition.json'),
        candidate_exported=False,platform_best=.48,
        note='Two new-head seeds share seed42 independent proxy backbones. Actual048 encoder was trained on all source records. No hidden-test claim.')
    (WORK/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'checks/presence_rescue_20261024.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 0.48之后：0/1读出与独立信号分支实验','',
        '已完成4个留房间划分 × 2个新分支随机种子 × 3组对照，共24次训练。每次固定40轮，没有用外层房间挑轮数或阈值。',
        '','A_actual使用实际底座的既有证据，底座曾见过所有源记录，只作为诊断。A_proxy改用此前仅在训练侧拟合的主干；两个种子针对新读出，不代表两次独立主干训练。B_signal不使用训练过的主干，直接读取确定性CSI预处理的静态幅度形状、动态幅度/相位以及绝对强度、时间统计。',
        '','所有划分先分完整采集记录，unknown不参与。复用之前固定的fit/calibration/外层房间划分。训练仅用真实0/1；阈值固定为max(0.5，训练侧校准空房最高预测概率+1e-6)，没有在外层房间搜索阈值。每折仅1条空房校准记录，无法证明未来误报率。',
        '','## 已有头诊断','',
        '实际0.48剩余134个1→0窗口中，原始视角直接头支持1人的有78个，序数头80个，静态专家59个；presence“有人”概率中位数约0.713。已有表征并非完全没有人数证据。单独抬高1人仍可能损害空房识别。',
        '','诊断使用冻结原权重在缓存特征上的PyTorch重建：960窗原始视角类别与实际ONNX一致，但最大概率差0.004755；四个最差样本CPU重建也有同样偏差。因此只用于分解，不能声称是ONNX逐数值等价验证。',
        '','## 独立主干代理管线','',
        '以下合计a1/c1/g1/e1四个互斥外层房间，共880窗，其中200个空房、300个1人。其他人数窗口仍保留在门控后评估。只有底座预测0时允许改1。',
        '','| 方法 | 新头种子 | 纠正/新增错误 | 救回1人 | 新增空房误报 | 空房召回下降 | 净改善房间 | 通过 |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for kind in ('A_actual_diagnostic','A_proxy','B_signal'):
        for seed in (42,2026):
            r=groups[kind][str(seed)]['independent_backbone_proxy']
            lines.append(f"| {kind} | {seed} | {r['corrected']}/{r['broken']} | {r['rescued_one']} | {r['added_empty_false_positive']} | {r['empty_recall_drop']:.2%} | {','.join(r['rooms_with_net_gain']) or '无'} | {'是' if r['passed'] and kind!='A_actual_diagnostic' else '否'} |")
    lines+=['','A_actual这一行使用见过全部标签的编码证据，即使与独立代理底座组合也不是独立管线，仅作泄漏风险对照；筛选只依据A_proxy与B_signal。','',
        '## 固定实际0.48门控的诊断','',
        '这里不是独立整管线测试，因为0.48底座见过全部源记录。原先880窗中实际0.48正确595窗。表中数字仅衡量新规则在这些源数据上的收益/代价。','',
        '| 方法 | 种子 | 纠正/新增错误 | 空房误报增加 | 净改善房间 |','|---|---:|---:|---:|---|']
    for kind in groups:
        for seed in (42,2026):
            r=groups[kind][str(seed)]['actual048']
            lines.append(f"| {kind} | {seed} | {r['corrected']}/{r['broken']} | {r['added_empty_false_positive']} | {','.join(r['rooms_with_net_gain']) or '无'} |")
    lines+=['','## 分支自身表现与解释','',
        '| 分支 | 种子 | 独立0/1分支正确/500 | 1人召回 | 空房召回 |',
        '|---|---:|---:|---:|---:|']
    for kind in ('A_proxy','B_signal'):
        for seed in (42,2026):
            r=groups[kind][str(seed)]['independent_backbone_proxy']
            lines.append(f"| {kind} | {seed} | {r['standalone_binary_correct']}/500 | {r['standalone_one_recall']:.2%} | {r['standalone_empty_recall']:.2%} |")
    lines+=['','多个校准空房概率接近1，安全阈值因此接近或超过1，分支无法在留出房间稳定兼顾检出与误报。这不是通过放宽阈值就能证明解决的问题。固定0.48读出的g1收益，同时伴随a1/e1大量空房误报，提示房间相关混淆仍然存在。',
        '','这些结果只否定本轮固定配方，不证明所有0/1分支都无效，也不证明原始信号缺乏信息。代理底座与实际底座性能差异较大，因此代理只作风险筛选，不能把它的提升直接换算成实际0.48的提升。',
        '','下一步可检验同房间0/1的成对差异监督，配合只在训练侧实际共享类别的跨房间约束；先证明对g1、a1、e1的迁移，再考虑融入主干。校准优先比较源记录交叉拟合，减少依赖单条空房记录。部署不能假定老师额外提供同房间空房参考。本轮尚未运行这些后续方案。',
        '','## 筛选决定','',
        f"A_proxy两个种子均通过：{eligibility['A_proxy']}；B_signal两个种子均通过：{eligibility['B_signal']}。",
        '筛选要求：两个种子下空房新增误报占空房总数不超过1%，至少两个房间有净收益，整体纠正多于破坏。不能把底座原本已错的空房排除出分母来掩盖代价。',
        '','本轮A_proxy、B_signal均未通过，因此没有导出新提交模型；继续保留实际平台0.48为最佳。本轮没有进行新ONNX接口/运行时间检测，因为没有选出导出候选。',
        '','所有权重、逐记录结果、阈值、训练损失、完整划分与预测保存在 `D:/huawei2026-experiments/presence_rescue_20261024/`。平台最好成绩不因本地实验自动改写。']
    # Compute reference count instead of trusting a manually copied value.
    total=groups['A_proxy']['42']['actual048']['baseline_correct']
    lines=[line.replace('实际0.48正确595窗',f'实际0.48正确{total}窗') for line in lines]
    (ROOT/'docs/PRESENCE_RESCUE_20261024.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(eligible=eligibility,summary={k:{s:{r:{v:groups[k][s][r][v] for v in ('corrected','broken','empty_recall_drop','rooms_with_net_gain')} for r in ('actual048','independent_backbone_proxy')} for s in ('42','2026')} for k in groups}),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
