"""Report all predeclared results, including failures; do not select a winning test seed."""
import shutil
from train_chain import *

def main():
    torch.set_num_threads(4);allresults={};aggregates={};records={}
    for variant in ('full','no_context','no_spread'):
      for seed in ((42,2026) if variant=='full' else (42,)):
        for family,folds in [('room',ROOMS),('record',FOLDS[4:])]:
            cm=torch.zeros(5,5,dtype=torch.long);lastcm=cm.clone();rec=[]
            for fold in folds:
                folder=WORK/f'{fold}_{variant}_s{seed}';r=old.read(folder/'result.json');cp=torch.load(folder/'final.pt',weights_only=True);assert old.sha(folder/'final.pt')==r['sha256'];a,b,c=[set(cp[k]) for k in ('fit_records','calibration_records','test_records')];assert not(a&b or a&c or b&c)
                d=torch.load(folder/'predictions.pt',weights_only=True);ix=d['test_indices'];y=d['labels'];pred=d['logits'].argmax(-1);cm+=torch.tensor(r['test']['confusion']);lastcm+=torch.tensor(r['last_test']['confusion']);allresults[folder.name]=r
                for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
                    jj=[i for i in ix if d['rows'][i]['identifier']==ident];votes=torch.bincount(pred[jj],minlength=5);rec.append(dict(fold=fold,identifier=ident,correct=int((pred[jj]==y[jj]).sum()),total=len(jj),majority_agreement=float(votes.max()/len(jj))))
            key=f'{family}_{variant}_s{seed}';records[key]=rec;aggregates[key]=dict(correct=int(cm.trace()),total=int(cm.sum()),confusion=cm.tolist(),last_correct=int(lastcm.trace()),last_confusion=lastcm.tolist(),macro_recall=float((cm.diagonal()/cm.sum(-1).clamp_min(1)).mean()),mean_record_agreement=float(np.mean([v['majority_agreement'] for v in rec])))
    rng=np.random.default_rng(20261028);ci={}
    for family in ('room','record'):
      full=records[f'{family}_full_s42']
      for variant in ('no_context','no_spread'):
        control=records[f'{family}_{variant}_s42'];assert [x['identifier'] for x in full]==[x['identifier'] for x in control]
        values=[]
        for _ in range(5000):
            num=den=0
            for f in sorted({x['fold'] for x in full}):
                group=[i for i,x in enumerate(full) if x['fold']==f]
                for i in rng.choice(group,len(group),replace=True):num+=full[i]['correct']-control[i]['correct'];den+=full[i]['total']
            values.append(num/den)
        ci[f'{family}_full_minus_{variant}']=np.percentile(values,[2.5,97.5]).tolist()
    source={}
    for seed in (42,2026):
        d=torch.load(WORK/f'actual_full_s{seed}'/'predictions.pt',weights_only=True);r=old.read(WORK/f'actual_full_s{seed}'/'result.json');source[str(seed)]=dict(selected=r['test'],last15=r['last_test'],epoch=r['epoch'])
    audit=dict(aggregates=aggregates,source=source,ablations_ci95=ci,development_fits=28,full_data_fits=2,platform_score=None,code_sha256={p.name:old.sha(p) for p in Path(__file__).parent.glob('*.py')})
    old.save(WORK/'research_results.json',audit);selection=old.read(WORK/'selection.json');published=old.read(WORK/'published.json');last=old.read(WORK/'published_last.json') if (WORK/'published_last.json').exists() else None
    lines=['# 联合证据链新模型：2026-10-28','',f'已完成28组开发训练和2组全数据训练，共450轮。每个模型186,554参数。主模型完整结构与种子42预先确定，源校准选择的全数据训练轮数为{selection["epochs"]}；另外保留固定15轮对照。平台未测，原平台最佳0.54继续保留。','',
    '## 识别链的实际实现','','原始输入使用已核验的确定性预处理得到状态、变化、天线对相位、ATI派生时间与强度摘要。新网络从随机初始化共同训练以下环节：稳定频率状态编码→局部变化/相位关系编码→尺度和状态上下文→有界上下文调节→30个局部时间片上的时间卷积→均值、离散程度、峰值和局部概率分歧→联合五类判断。没有复用旧037/054学习权重，没有冻结旧主干再加纠错器。','',
    '局部概率只是一种可学习中间表示，没有真实局部人数/活动监督，不能称为可解释的个人检测或校准不确定性。本版沿用旧预处理，尚未新增原始有效载波质量通路或恢复被平均的原始细节。此版本是整条思路的首个可运行近似，不代表所有假设已实现或验证。','',
    '## 训练与公平对照','','只使用48条比赛有标签记录的960个非重叠窗口；正反时间两视图。AdamW学习率0.0006、权重衰减0.02，15轮余弦计划，batch24窗口，NLL+.05时间反转一致性。按人数→房间→记录→窗口均衡抽样。无U、外部数据、伪标签、输出偏置扫描或硬保护规则。','',
    '完整模型两个种子42/2026；no_context与no_spread各种子42。三个结构参数数量相同、初始化种子和抽样流程匹配，但禁用分支的有效容量不同。no_context只关闭局部上下文调节，仍保留末端上下文；no_spread关闭时间标准差、峰值相对均值信息和局部概率离散程度，仍保留均值。','',
    '四个旧留房间折、三个整记录分组折；训练、校准、测试按整条记录隔离。源校准最低均衡NLL选择权重，不使用目标标签选epoch；完整模型14个源选择轮数的中位数决定全数据模型轮数。全量训练仍跑满15轮并保留last权重。所有折都属于反复使用的开发数据，整记录不保证每个测试房间都出现在训练中。','',
    '## 结果','','|协议|结构/种子|源校准选中权重|固定15轮|类别平均召回|记录内多数预测占比|','|---|---|---:|---:|---:|---:|']
    for key,a in aggregates.items():lines.append(f"|{key.split('_')[0]}|{key.split('_',1)[1]}|{a['correct']}/{a['total']} ({a['correct']/a['total']:.2%})|{a['last_correct']}/{a['total']}|{a['macro_recall']:.2%}|{a['mean_record_agreement']:.2%}|")
    lines+=['','room为整房间、record为整记录。多数预测占比高可能只是恒定预测，必须结合类别召回；不能独立当作模型进步。旧054重训代理在同四房间为278/880，仅供历史对照，训练配置与本轮不同。平台0.54不能与这些本地百分比直接相减。','',
    '## 两个新连接的消融','','|差值：完整－消融|按整记录bootstrap 95%区间|','|---|---|']
    for k,v in ci.items():lines.append(f'|{k}|{v}|')
    lines+=['','区间按折内完整记录重采样5000次；只覆盖当前记录采样的不确定性，没有包含全部模型搜索及训练随机性。消融只有一个种子，不能据此宣称已证明机制优越。','',
    '实际解释：seed42完整模型留房间230/880，高于去上下文198和去时间离散信息190；但整记录422/960低于去上下文440，高于去离散信息406。完整模型另一种子留房间仅149/880。四个消融差值区间均包含零，当前没有稳定证据证明两个连接有效。不能将点估计差值直接归因于某个物理机制，因为各自源校准选中的训练轮数也不同。','',
    '固定15轮完整模型整记录为491/960和493/960，但两个种子留房间都只有70/880；去时间离散信息对照的固定15轮整记录反而为503/960。这排除了“训练久一些就能证明识别链正确”的解释。seed42源选择权重的留房间0人和4人召回均为0，核心问题仍是环境变化后证据与人数的关系失效。此轮没有额外标签证明网络识别了静止的人、活动或房间属性。','',
    '## 完整模型逐房间结果','','|房间|seed42|seed2026|','|---|---:|---:|']
    for f in ROOMS:lines.append(f"|{f}|{allresults[f'{f}_full_s42']['test']['correct']}/{allresults[f'{f}_full_s42']['test']['total']}|{allresults[f'{f}_full_s2026']['test']['correct']}/{allresults[f'{f}_full_s2026']['test']['total']}|")
    lines+=['','## 全数据模型：只能解释为训练拟合','','|种子|源选择轮数|选中权重正确/960|固定15轮正确/960|','|---|---:|---:|---:|']
    for seed,v in source.items():lines.append(f"|{seed}|{v['epoch']}|{v['selected']['correct']}/960|{v['last15']['correct']}/960|")
    lines+=['','## 交付与边界','',f"主研究ONNX：`{published['path']}`，{published['bytes']:,}字节，SHA256 `{published['sha256']}`。",f"CPU资源检查：{published['resource_check']['status']}，最慢测试样例平均{max(c['cpu_mean_s'] for c in published['resource_check']['cases']):.4f}秒。"]
    if last:lines.append(f"固定15轮研究对照：`{last['path']}`，{last['bytes']:,}字节，资源检查{last['resource_check']['status']}。")
    lines+=['','两份都是未评分研究模型，没有推荐替换已测054。同时保存两个种子的PyTorch权重，及每个开发折的最佳/第15轮权重和预测。数值/资源检查通过不代表平台准确率或平台下载一定成功。','',
    '后续围绕两个连接的消融和各类错误推进：完整结构若不优于消融，应修正对应假设；训练拟合高而跨房间低，则不能靠更多epoch宣称解决泛化。保留初期低分结果，但不把低分解释为未来必然成功。','',
    '代码：`experiments/evidence_chain_20261028/`；完整实验：`D:/huawei2026-experiments/evidence_chain_20261028/`。已核验28个开发模型的权重hash和记录隔离，检查前后通道置换误差，ONNX用原始CSI窗口与缓存/PyTorch对照并进行独立CPU检查。']
    path=ROOT/'docs/EVIDENCE_CHAIN_20261028.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(path,ROOT/'2026-10-25_037配方与种子复现/联合证据链新模型训练_20261028.md');print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
