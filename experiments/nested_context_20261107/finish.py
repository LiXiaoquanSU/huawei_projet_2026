"""Record all completed controls without changing the platform best."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 sampling=read(Path('D:/huawei2026-experiments/context_symmetry_20261107_class_sampling/results.json'));trees=read(HERE/'tree_probe/results.json')
 assert len(sampling)==9 and len(trees['results'])==9
 totals={}
 for r in sampling:totals.setdefault('class_sampling/'+r['mode'],[]).append(r['metrics']['correct'])
 for r in trees['results']:totals.setdefault(r['method'],[]).append(r['correct'])
 summary=dict(date='2026-11-07',new_neural_fits=27,new_neural_epochs=135,new_tree_fits=9,platform_best=.54,next_submission=None,platform_score=None,totals={k:dict(per_fold=v,correct=sum(v),total=960) for k,v in totals.items()},status='completed_negative_controls_no_export',reports=['docs/NESTED_CONTEXT_20261107.md','docs/CONTEXT_SYMMETRY_20261107.md','docs/RESEARCH_ROUND_20261107.md'])
 (HERE/'completed.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['# 2026-11-07 实验执行总结','','已完成 27 次神经网络训练（135 轮）及 9 次树模型训练，另完成源 OOF 校准与融合。平台最佳保持 0.54，没有新的平台成绩；本轮没有导出新 ONNX，也没有宣称达到 0.60。','','## 完成的四项检验','','1. 原037与上下文039结构的三折嵌套记录交叉拟合：18 次训练。固定5轮上下文477/960，原结构422/960；源侧自动选轮未带来稳定提升。OOF 校准后的上下文461/960，简单融合427/960，学习选择器408/960（固定5轮）。不支持这次融合设置。','2. 上下文的变换增强：6 次新增训练，原训练权重复用。原训练＋双时间推理480/960；时间反转训练＋双时间推理473/960；再加Rx排列训练＋12视角推理435/960。当前5轮预算下未改善；不能推论任何预算下都无用。','3. 只按人数等权采样：3 次新增训练，对比原人数×房间采样。保持结构、seed42、5轮、每轮样本数和优化器不变。','4. 固定信号统计量＋树模型：3折×3模型。幅度/相位按32频率区间保留均值、标准差、相邻变化及lag4/16相关性，另保留17维摘要。ExtraTrees固定300树、深10、叶10；Boost固定120步、7叶、L2=10；均未用外层标签调参。','','|后两项方法|各折正确数/320|正确/960|准确率|','|---|---|---:|---:|']
 for k,v in totals.items():lines.append(f'|{k}|{v}|{sum(v)}|{sum(v)/960:.2%}|')
 lines+=['','## 学到什么','','- 本次原训练双时间推理的480/960是本地记录折结果，不能和平台054直接比较。原054权重训练过全部48条记录，未拿它在这些记录上的分数伪装成留出结果。','- OOF中两个专家有不同错误，不代表一个小选择器能正确挑专家。校准和门控在本次设置下反而损失准确率。','- 房间等权确实改变训练分布：1人g1约55.6%变为14.3%，三条unknown合计约16.7%变为42.9%。但取消房间平衡也未改善，不能只凭权重变化宣布找到了原因。','- 这些树模型对照未显示“固定统计量已经足够，只要更换读出器就能提高”的证据。这里只检验了指定特征与模型，不排除其他统计方法。','- 27次拟合是27套实验训练，不是27个有效提交候选；同一960窗口跨实验复用也不增加独立验证样本。','','## 当前决定与限制','','保留原054为已测最佳，保存所有实验权重及失败记录；不把未经证实的新模型打包成推荐提交。此次未找到有足够提升依据的新候选，0.60目标尚未完成。','接下来需要新的可检验信息来源或更可靠的泛化证据，再启动完整训练；不根据这批开发折继续大规模挑种子/偏置，也不把同一记录的密集窗口当独立测试。','','文件：`experiments/nested_context_20261107/`；大型神经权重均在D盘。完整混淆矩阵、逐折预测和固定协议分别保存在各实验结果中。']
 (ROOT/'docs/RESEARCH_ROUND_20261107.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 registry=read(ROOT/'models/registry.json');before=registry.get('current_best');registry['research_round_20261107']=summary;registry['updated_date']='2026-11-07';registry['research_status']='nested_context_symmetry_sampling_and_tree_controls_complete_no_export'
 assert registry.get('current_best')==before
 (ROOT/'models/registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');prefix='当前研究（2026-11-07）：完成27次神经网络训练（135轮）及9次树模型对照；源交叉拟合融合、变换增强、采样修改均未形成新的提交依据。本轮上下文对照双时间推理480/960（本地开发记录折，非平台成绩）。已测最佳仍0.54，暂无新的推荐提交文件。见[本轮完整总结](docs/RESEARCH_ROUND_20261107.md)。以下为历史。\n\n'
 if not text.startswith(prefix):p.write_text(prefix+text,encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
