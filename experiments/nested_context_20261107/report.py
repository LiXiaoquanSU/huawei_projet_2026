import json
from pathlib import Path
import numpy as np
import torch
W=Path('D:/huawei2026-experiments/nested_context_20261107')
ROOT=Path(__file__).resolve().parents[2]
def main():
 r=json.loads((W/'results.json').read_text())['results'];f=json.loads((Path(__file__).parent/'fusion_results.json').read_text())
 lines=['# 2026-11-07：源记录交叉拟合与上下文融合复盘','','平台最佳仍为 0.54；本报告所有数字都是反复使用的本地开发记录折，不能当作平台成绩。没有生成或推荐新的提交文件。','','## 方法','','3 个外层记录折，每折留出 16 条记录（320 窗口），其余 32 条记录再做两折交叉拟合。两个结构均从 seed42 初始化、AdamW、5 轮、原 physical 损失、人数×房间平衡采样，每轮抽取源窗口数个样本。共 18 次拟合。每条交叉预测对应的模型未训练该记录；外层留出记录不参与选轮、偏置或选择器拟合。','','选轮依赖源 OOF 的宏准确率或平衡 NLL；固定第 5 轮作为对照。OOF 选轮后用于拟合校准器，仍存在源侧模型选择复用；外层评估隔离。内层训练规模仅约 16 条记录，模型置信度及最优轮数未必迁移到 32 条记录。','','## 未校准单模型','','|规则|结构|正确/960|准确率|各折正确数/320|','|---|---|---:|---:|---|']
 for criterion in ['accuracy','nll','fixed5']:
  for v in ['baseline','context_events']:
   a=[x for x in r if x['criterion']==criterion and x['variant']==v];counts=[x['metrics']['correct'] for x in a];n=sum(counts)
   lines.append(f'|{criterion}|{v}|{n}|{n/960:.2%}|{counts}|')
 lines+=['','## 源 OOF 校准与融合','','固定 L2 温度/偏置校准，以源 OOF 常规交叉熵拟合；不使用外层标签。选择器是固定 C=0.1 的逻辑回归，用两模型概率、熵、间隔和差异判断两个专家谁更可能正确，只在源 OOF 恰有一方正确的样本上训练；融合比例限制在 0.25–0.75。无房间或记录 ID。','','|选轮|方式|正确/960|准确率|','|---|---|---:|---:|']
 for k,n in f['totals'].items():
  a,b=k.split('/');lines.append(f'|{a}|{b}|{n}|{n/960:.2%}|')
 lines+=['','简单融合和所测试的选择器均低于上下文模型单独使用。发现专家错题互补，并不意味着能在未知样本上判断应该信谁。此结果不支持导出融合候选，也不能证明任何融合都无效。','','## 下一项已启动的检验','','保持上下文结构、seed42、固定 5 轮、相同优化器步数与采样预算，对比：原输入；随机时间方向；随机时间方向加接收通道排列。时间反转使用重新预处理缓存，接收排列同步修改幅度、相位配对及共轭符号、摘要；每批重新计算对应信号辅助目标。评价原视角、双时间视角、12 视角平均概率；不以外层分数选择训练轮数。该测试只检验这一结构上的变换增强，并非首次尝试所有架构上的排列增强。','','训练输出：`D:/huawei2026-experiments/context_symmetry_20261107/`。未完成前不预报提升。']
 (ROOT/'docs/NESTED_CONTEXT_20261107.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\n'.join(lines[10:24]))
if __name__=='__main__':main()
