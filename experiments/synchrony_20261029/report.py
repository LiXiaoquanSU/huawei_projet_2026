"""Aggregate record-cluster uncertainty and empty-room controls."""
import json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=Path('D:/huawei2026-experiments/synchrony_20261029')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def bootstrap(groups,seed=29):
    rng=np.random.default_rng(seed);values=[]
    for _ in range(3000):
        vals=[]
        for g in groups:vals.extend(rng.choice(g,size=len(g),replace=True).tolist())
        values.append(np.mean(vals))
    return np.percentile(values,[2.5,97.5]).tolist()
def main():
    summaries=[];count=[];differences=[]
    for control,path in [('original',BASE),('frequency_centered',BASE/'spectral_control')]:
        d=read(path/'results.json');pred=read(path/'predictions.json');rows=d['rows'];y=np.array([r['headcount'] for r in rows])
        for family,folds in [('record',('record0','record1','record2')),('room',('a1','c1','g1','e1'))]:
            for mode in ('rx','amp_phase'):
              for neg in ('shift','other_window'):
                rr=[r for r in d['results'] if r['task']=='synchrony' and r['fold'] in folds and r['mode']==mode and r['negative']==neg]
                groups=[]
                for f in folds:
                    z=pred[f'{f}_{mode}_{neg}'];ix=np.array(z['indices']);p=np.array(z['probabilities']).argmax(1);n=len(ix);scores=((p[:n]==1).astype(float)+(p[n:]==0))*.5
                    groups.append(np.array([scores[[rows[i]['identifier']==ident for i in ix]].mean() for ident in sorted({rows[i]['identifier'] for i in ix})]))
                summaries.append(dict(control=control,family=family,mode=mode,negative=neg,correct=sum(r['correct'] for r in rr),total=sum(r['total'] for r in rr),ci95=bootstrap(groups),per_count={str(k):dict(correct=sum(r['per_count'][str(k)]['correct'] for r in rr),total=sum(r['per_count'][str(k)]['total'] for r in rr)) for k in range(5)}))
            for mode in ('summary','relation','combined'):
                rr=[r for r in d['results'] if r['task']=='count' and r['fold'] in folds and r['mode']==mode];cm=np.sum([r['confusion'] for r in rr],axis=0)
                count.append(dict(control=control,family=family,mode=mode,correct=int(cm.trace()),total=int(cm.sum()),confusion=cm.tolist()))
                if mode!='summary':
                    groups=[]
                    for f in folds:
                        z=pred[f'{f}_count_{mode}'];ref=pred[f'{f}_count_summary'];assert z['indices']==ref['indices'];ix=np.array(z['indices'])
                        delta=(np.array(z['probabilities']).argmax(1)==y[ix]).astype(float)-(np.array(ref['probabilities']).argmax(1)==y[ix]).astype(float)
                        groups.append(np.array([delta[[rows[i]['identifier']==ident for i in ix]].mean() for ident in sorted({rows[i]['identifier'] for i in ix})]))
                    differences.append(dict(control=control,family=family,mode=mode,mean_delta=float(np.concatenate(groups).mean()),ci95=bootstrap(groups)))
    audit=dict(synchrony=summaries,count=count,count_difference_ci=differences,trained_logistic_fits=294,platform_score=None)
    (BASE/'summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 跨天线同步是否是人数证据：2026-10-29','','## 问题与方法','','上轮天线图早期权重几乎不依赖端点绑定。本轮先验证这种对应关系是否在当前模型可访问的缓存特征中可检测，再验证它是否具有人数相关性，而不是先把同步判别强加给人数模型。','',
    '使用全部48条标注记录、960个非重叠窗口。幅度先对齐到32个频率组，与相位组对应。对三对天线，在三个时间尺度计算相关系数、绝对相关、频率间离散程度和能量轨迹相关，跨天线对对称汇总为48维。幅度—相位组使用幅度变化能量与标准化相位特征变化能量，不声称是原始物理相位或绝对人体运动。','',
    '正样本：同窗口同步。负样本1：裁剪相差29个缓存时间点的片段，不循环拼接；负样本2：使用同记录相隔7个非重叠窗口的另一信号。每种二分类正负数相等，机会水平50%。每条记录所有正负样本都留在同一个训练/校准/测试分区，保持房间和人数标签相同。','',
    '控制：从每根天线、每个时刻减去它自己的跨频率均值，去掉宽频共同变化；不在三根天线之间相互减均值，避免人为制造接收通道负相关。该操作也可能去掉人体宽频变化，不能当成纯设备噪声隔离。相位特征与人数摘要未改。','',
    '三折整记录、四折留房间，标准化只拟合训练部分。逻辑回归C∈{0.01,0.1,1}仅由源校准均衡NLL选定。两组控制共294次轻量拟合。同步判别与人数判别是不同任务，不能把前者90%以上解释为人数识别成功。','',
    '## 同步判别','','|控制|协议|证据|错配方式|同步判别正确率|空房判别正确率|按整记录重采样95%区间|','|---|---|---|---|---:|---:|---|']
    for r in summaries:
        e=r['per_count']['0'];lines.append(f"|{r['control']}|{r['family']}|{r['mode']}|{r['negative']}|{r['correct']/r['total']:.2%}|{e['correct']/e['total']:.2%}|{[round(x,4) for x in r['ci95']]}|")
    lines+=['','record为整记录隔离，room为整房间隔离。空房的同步也可以来自共享硬件、环境或预处理产生的相关性；本实验没有标定信息来分解这些来源。','',
    '## 同步特征用于人数分类','','|控制|协议|输入|正确数/窗口数|正确率|','|---|---|---|---:|---:|']
    for r in count:lines.append(f"|{r['control']}|{r['family']}|{r['mode']}|{r['correct']}/{r['total']}|{r['correct']/r['total']:.2%}|")
    lines+=['','summary为13维接收通道对称强度/时间摘要；relation为96维真实同步特征；combined为二者拼接。人数探针只输入真实同步样本，未将错配样本伪装成新的标注人数训练数据。','',
    '|相对摘要基线|协议|特征|准确率差|按整记录重采样95%区间|','|---|---|---|---:|---|']
    for r in differences:lines.append(f"|{r['control']}|{r['family']}|{r['mode']}|{r['mean_delta']:+.2%}|{[round(x,4) for x in r['ci95']]}|")
    lines+=['','区间按各折内完整记录重采样3000次，不能覆盖反复架构探索的选择偏差、全部随机种子或未知测试分布。没有新增盲测数据。','',
    '## 结论与后续选择','','1. 当前预处理后的同步关系可被可靠检测，并非“数据完全没有对应信息”。','',
    '2. 空房中同步判别同样很高，因此同步不是人体特有证据。去掉跨频率共同变化也没有消除这一现象；不能简单归因于宽频增益，更不能宣称已分离人体与硬件。','',
    '3. 人数探针表现远低于同步判别，且跨房间收益不稳定。同步可能适合检查信号一致性，但本轮没有证明“训练同步辅助任务”能改进人数识别。也不能据轻量线性探针排除更复杂网络利用这些特征的可能。','',
    '4. 下一步应研究同一房间内部哪些变化随人数具有可重复关系。a1与c1共同覆盖1—4人，但c1没有空房，不能虚构统一空房基线；可先比较两房间1→2→3→4的变化方向，空房/有人另用具有0人数据的记录分析。带已知人数的房间参考只能用于机制诊断，不能偷偷给提交模型提供测试房间基线。','',
    '本轮没有新ONNX或平台提交。原模型完整保留。原始窗口、缓存或数据划分没有修改。代码：`experiments/synchrony_20261029/`；结果及逐样本概率：`D:/huawei2026-experiments/synchrony_20261029/`，其中`spectral_control/`保存共同变化控制。']
    p=ROOT/'docs/SYNCHRONY_20261029.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/同步关系机制验证_20261029.md')
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
