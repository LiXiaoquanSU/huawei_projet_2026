"""Integrated research report: semantic losses and genuinely additional signal information."""
import sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/count_order_20261029'))
from report import evaluate,old
BASE=Path('D:/huawei2026-experiments');WORK=BASE/'static_texture_20261029'

def main():
    summary={};runs={};comparisons={};rng=np.random.default_rng(29)
    configs=[('texture','texture_chain_20261028','full'),('order','count_order_20261029','full'),('identity','count_identity_20261029','full'),('static_phase','static_texture_20261029','full'),('zero_control','static_texture_20261029','zero')]
    for name,folder,variant in configs:
      for cp in ('selected','last'):
        rr=[evaluate(BASE/folder/f'record{i}_{variant}_s42',cp) for i in range(3)];key=f'{name}_{cp}';runs[key]=rr
        summary[key]=dict(correct=sum(r['correct'] for r in rr),total=sum(r['total'] for r in rr),mae=float(np.mean([r['mae'] for r in rr])),epochs=[r['epoch'] for r in rr],rank_correct=sum(r['rank_correct'] for r in rr),rank_total=sum(r['rank_total'] for r in rr),confusion=np.sum([r['confusion'] for r in rr],axis=0).tolist())
    for cp in ('selected','last'):
      for new,base in [('order','texture'),('identity','texture'),('static_phase','zero_control'),('static_phase','texture')]:
        groups=[]
        for a,b in zip(runs[f'{new}_{cp}'],runs[f'{base}_{cp}']):
            assert [r['identifier'] for r in a['records']]==[r['identifier'] for r in b['records']]
            groups.append(np.array([(x['correct']-y['correct'])/x['total'] for x,y in zip(a['records'],b['records'])]))
        draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)]
        comparisons[f'{new}_minus_{base}_{cp}']=dict(delta=float(np.concatenate(groups).mean()),ci95=np.percentile(draws,[2.5,97.5]).tolist())
    probe=old.read(BASE/'static_phase_20261029/probe_results.json');probes=[]
    for family,folds in [('record',('record0','record1','record2')),('room',('a1','c1','g1','e1'))]:
      for mode in ('summary','curvature','combined'):
       for model in ('linear','rbf'):
        rr=[r for r in probe['results'] if r['fold'] in folds and r['mode']==mode and r['model']==model];probes.append(dict(family=family,mode=mode,model=model,correct=sum(r['correct'] for r in rr),total=sum(r['total'] for r in rr)))
    old.save(WORK/'summary.json',dict(summary=summary,comparisons=comparisons,probes=probes,platform_score=None,new_neural_fits=12,new_training_epochs=240,static_probe_fits=126))
    lines=['# 人数监督与静态相位证据：2026-10-29','','本轮新增12组神经网络训练、240轮，18项记录级趋势诊断，以及126次静态相位轻量探针拟合。目标是合理可检验的识别链，0.6不是通过换指标或选择测试窗口凑出的数字。所有结果来自重复使用的开发划分，不等于平台成绩。','',
    '## 三个问题','','1. 同房间人数递增是否有简单共同规律？a1与c1的运动能量、高时间频率比例与人数相关方向相反，不支持“人数更多→变化更强”的统一规则。只有11/12条对应记录，不能作因果结论。','',
    '2. 是否能通过人数语义监督，减少环境捷径？分别测试同房间跨记录顺序损失，以及同人数跨房间正配对/不同人数同房间负配对。两者各自只在原纹理版上增加一个训练目标，未混合。未知房间不当作可确认的跨房间正样本。','',
    '3. 是否缺少静态空间相位信息？原相位特征按时间去均值。新增原始CSI的相对相位频率曲率，消除理想整体/线性频率相位歧义，保留稳定相位形状。它也可能反映房间与硬件，不能预设为人体特有。','',
    '## 神经网络对照','','|方案/权重|正确/960|期望人数MAE|记录对排序正确/总数|训练轮数|','|---|---:|---:|---:|---|']
    for k,r in summary.items():lines.append(f"|{k}|{r['correct']}/960 ({r['correct']/960:.2%})|{r['mae']:.3f}|{r['rank_correct']}/{r['rank_total']}|{r['epochs']}|")
    lines+=['','排序率不是人数正确率。selected由源校准均衡NLL选择；last固定20轮。static_phase与zero_control同为5,761,925参数、同初始化、同训练设置；差异只是新增384维信息是否置零。零分支虽参数相同，但有效容量不同。','',
    '|差值|窗口正确率差|整记录bootstrap95%区间|','|---|---:|---|']
    for k,r in comparisons.items():lines.append(f"|{k}|{r['delta']:+.2%}|{[round(v,4) for v in r['ci95']]}|")
    lines+=['','每折内以整记录重采样3000次，不能覆盖全部结构选择和训练种子不确定性。除明确的训练目标/静态信息改动外，使用相同源数据、三折整记录划分和种子42；没有测试标签进入优化、预处理或特征选择。','',
    '## 静态相位轻量探针','','|协议|输入|模型|正确/总数|','|---|---|---|---:|']
    for r in probes:lines.append(f"|{r['family']}|{r['mode']}|{r['model']}|{r['correct']}/{r['total']}|")
    checks=old.read(BASE/'static_phase_20261029/protocol.json')['checks']
    lines+=['',f'原始特征合成检查误差：{checks}；零输入有限。每个窗口独立提取，没有使用测试房间空房/已知人数参考。该不变性不是对所有硬件失真的免疫。','',
    '## 产物和限制','','本轮神经网络只完成整记录开发实验；对应留房间、全量训练及提交导出没有执行。现有平台0.54文件不变。所有selected/last权重、逐窗口预测、配对数量和loss已保存，可用于后续机制检查。','',
    '目录：`D:/huawei2026-experiments/count_order_20261029/`、`count_identity_20261029/`、`static_phase_20261029/`、`static_texture_20261029/`。代码位于项目experiments同名目录。']
    p=ROOT/'docs/COUNT_EVIDENCE_20261029.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/人数监督与静态相位实验_20261029.md');print(summary,comparisons,flush=True)
if __name__=='__main__':main()
