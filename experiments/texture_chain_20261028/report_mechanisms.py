"""Report structure evidence before leaderboard ambitions; no omitted unsuccessful fold."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];BASE=Path('D:/huawei2026-experiments');WORK=BASE/'mechanism_20261029'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    runs={};byfold=[];audit=read(WORK/'audit.json')['results'];probes=[]
    for name in ('stable_chain_20261028','texture_chain_20261028'):
        cm=np.zeros((5,5),dtype=int);last=cm.copy();fits=[]
        for f in ('record0','record1','record2'):
            p=BASE/name/f'{f}_full_s42';r=read(p/'result.json');cp=torch.load(p/'final.pt',weights_only=True)
            a,b,c=[set(cp[k]) for k in ('fit_records','calibration_records','test_records')];assert not(a&b or a&c or b&c)
            with (p/'final.pt').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==r['sha256']
            cm+=np.array(r['test']['confusion']);last+=np.array(r['last_test']['confusion']);fits.append(r['history'][-1]['fit'])
            byfold.append((name,f,r['epoch'],r['test']['correct'],r['last_test']['correct']))
        runs[name]=dict(correct=int(cm.trace()),total=int(cm.sum()),recall=(cm.diagonal()/cm.sum(1)).tolist(),confusion=cm.tolist(),last_correct=int(last.trace()),last_confusion=last.tolist(),last_fit= fits)
        for checkpoint in ('final','last'):
          for condition in ('original','coherent_rx','phase_assignment_only','phase_time_shift','static_other_window'):
            rows=[r for r in audit if r['architecture']==name and r['checkpoint']==checkpoint and r['condition']==condition]
            assert len(rows)==3
            probes.append(dict(architecture=name,checkpoint=checkpoint,condition=condition,total=sum(r['total'] for r in rows),correct=sum(r['correct'] for r in rows),changed_predictions=sum(r['changed_predictions'] for r in rows),mean_probability_L1=float(np.mean([r['mean_probability_L1'] for r in rows])),max_probability_error=max(r['max_probability_error'] for r in rows)))
    result=dict(runs=runs,probes=probes,platform_score=None,completed_fits=6,completed_epochs=150,no_new_submission=True)
    (WORK/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 识别链机制实验：2026-10-29','','本轮优先验证合理的识别链，60%只作为长期目标。完成两种结构各三折整记录训练，共6组、150轮，以及全部960个独立记录窗口上的依赖检查。尚未进行这两种结构的留房间或全数据训练，没有导出新提交文件。平台最佳0.54未改变。','',
    '## 两个实验回答的问题','','稳定融合版（1127万参数）：分支归一化、渐进残差交互、动态/状态辅助监督以及较低学习率和预热，能否改善训练并转移到独立记录？30轮。','',
    '二维纹理版（543万参数）：静态幅度、动态幅度和局部残差先共同进行二维卷积，再汇总天线，能否保留早期统计汇总丢掉的线索？另保留相位纹理和绝对尺度，20轮。该版没有幅度—相位端点绑定，是对应关系敏感性检查的结构性负对照。','',
    '两者沿用相同缓存、三折整记录划分、种子42和双时间视图。校准与测试记录独立，选epoch只看源校准均衡NLL。固定末轮另报告。分支归一化等多项修改同时进行，不能把效果归因于单一模块。旧预处理的平均操作不变，本轮未恢复原始CSI细节。','',
    '## 正常输入结果','','|模型|源校准选择|固定末轮|0—4人召回（源选）|','|---|---:|---:|---|']
    for name,r in runs.items():lines.append(f"|{name}|{r['correct']}/960 ({r['correct']/960:.2%})|{r['last_correct']}/960 ({r['last_correct']/960:.2%})|{[round(x,3) for x in r['recall']]}|")
    lines+=['','历史同划分：小版源选422/960、固定15轮491/960；大版源选405/960、固定15轮336/960。以上均是反复使用的开发数据，不能称盲测或平台成绩。','',
    '|模型/折|源选epoch|源选正确/320|末轮正确/320|','|---|---:|---:|---:|']
    for name,f,e,c,l in byfold:lines.append(f'|{name}/{f}|{e}|{c}|{l}|')
    lines+=['','## 模块依赖检查','','在每折测试记录上，分别评估源选和末轮权重。coherent_rx同时交换天线及正确相位关系，应严格不变；phase_assignment_only只交换相位边而不交换幅度节点；phase_time_shift将相位相对幅度错开30个缓存时间点，正反视图方向相反；static_other_window替换成同一记录另一窗口的静态特征。','',
    '后三种输入不一定物理自洽。它们只能证明网络依赖哪些信息，不能证明其学习了正确物理因果，也不能当成合法增强或准确率改进方法。静态替换仍维持记录标签，但不维持原窗口的完整信号对应。','',
    '|模型/权重|条件|预测改变/960|平均概率L1变化|正确/960（仅敏感性参考）|','|---|---|---:|---:|---:|']
    for r in probes:lines.append(f"|{r['architecture']}/{r['checkpoint']}|{r['condition']}|{r['changed_predictions']}|{r['mean_probability_L1']:.6f}|{r['correct']}|")
    lines+=['','结构性不变性来自共享参数及对称汇总，不应描述成模型学会了正确物理规律。机制结论必须结合输入干预的局限和未干预数据上的错误。','',
    '## 本轮可以支持的结论','','1. 稳定融合版末轮三折训练拟合为540/540、539/540、540/540，但独立记录只有429/960。已证明其具备拟合能力，尚未证明证据链可泛化。','',
    '2. 稳定融合版源选权重对相位端点错配的平均概率L1变化仅约0.000091，预测改变0/960；末轮为约0.02648、12/960。这支持“当前早期权重几乎不依赖端点绑定”的描述，不支持“相位整体没有作用”或“所有天线图都无用”。','',
    '3. 纹理版源选结果479/960，好于上轮大版405/960，但末轮388/960。这里的优势依赖权重阶段，不能把所有提升归因于二维纹理，也不能称已经找到正确物理模型。','',
    '4. 同记录替换静态特征在源选权重中影响很小，可能因为同记录背景相近；不能据此断言静态通路没有用。相位循环移位同样可能保留大量全局统计或引入边界效应。','',
    '下一步优先验证数据是否提供可恢复的跨天线同步/对应证据，再考虑训练这种关系：先用同记录、相同人数的错配对照排除房间和类别捷径；只有真实对应能被稳定区分，才考虑加入相应辅助任务。不能为了让图模块“有反应”而强迫网络学习未证实有用的差异。','',
    '审计程序曾因默认GPU计算精度与训练不一致而触发原预测复现断言；统一TF32/确定性设置和batch后，所有12份权重通过原预测复现与一致天线置换检查。最终表格只使用通过检查的重跑结果。','',
    '## 保存位置','','训练：`D:/huawei2026-experiments/stable_chain_20261028/`、`D:/huawei2026-experiments/texture_chain_20261028/`。机制审计：`D:/huawei2026-experiments/mechanism_20261029/`。所有模型的selected/last权重、逐窗口预测和曲线保留。代码分别位于同名experiments目录。']
    p=ROOT/'docs/MECHANISM_CHAIN_20261029.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/识别链机制实验_20261029.md')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
