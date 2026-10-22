"""Summarize feedback, data quality, fixed room tests and real trained artifacts."""
import csv,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from train import ROOT,WORK,read,sha
from acquire import save
def main():
    audit=read(WORK/'cleaning_audit.json');verification=read(WORK/'verification.json');selection=read(WORK/'selection.json');screen=read(WORK/'screen_results.json');rep=read(WORK/'replicate_results.json');primary=read(WORK/'primary_candidate.json');models=read(WORK/'export_registry.json');expert=read(WORK/'expert_decision.json')
    fits=[read(p/'metrics.json') for p in (WORK/'fits').iterdir() if (p/'metrics.json').exists()];pre=[read(p/'metrics.json') for p in (WORK/'pretrain').iterdir() if (p/'metrics.json').exists()];full=[m for m in fits if m['config']['fold']=='full'];dev=[m for m in fits if m['config']['fold']!='full']
    coarse=WORK/'_coarse_frequency_v1';coarse_fits=list(coarse.glob('fits/*/metrics.json'));coarse_pre=list(coarse.glob('pretrain/*/metrics.json'))
    rows=screen+[r for r in rep if r['seed']!=42]
    table='\n'.join(f"|{r['variant']}|{r['seed']}|{r['room_mean']:.1%}|{r['macro_recall']:.1%}|{r['worst']:.1%}|" for r in rows)
    e1table='\n'.join(f"|{r['variant']}|{r['e1']['accuracy']:.1%}|{sum(sum(row[2:]) for row in r['e1']['confusion_matrix'][:2])}/60|" for r in screen)
    external_table='\n'.join(f"|{m['family']}|{m['seed']}|{m['external_validation']['accuracy']:.1%}|" for m in pre if m['external_validation'])
    fulltable='\n'.join(f"|{m['config']['variant']}|{m['config']['seed']}|{m['training_evaluation']['accuracy']:.1%}|{m['actual_records']}|{m['actual_windows']}|" for m in full)
    modeltable='\n'.join(f"|{m['path']}|{m['bytes']/1e6:.2f}MB|{max(c['cpu_mean_s'] for c in m['resource']['cases']):.3f}s|" for m in models)
    mean={v:dict(room_mean=float(np.mean([r['room_mean'] for r in rep if r['variant']==v])),macro=float(np.mean([r['macro_recall'] for r in rep if r['variant']==v]))) for v in ('source_transfer',selection['variant'])}
    class_matrices={v:np.asarray([r['confusion_matrix'] for r in rep if r['variant']==v]).sum(0) for v in ('source_transfer',selection['variant'])}
    recalls={v:np.diag(cm)/cm.sum(1) for v,cm in class_matrices.items()}
    recalltable='\n'.join(f"|{c}人|{recalls['source_transfer'][c]:.1%}|{recalls[selection['variant']][c]:.1%}|" for c in range(5))
    out=primary['resource'];maxcpu=max(c['cpu_mean_s'] for c in out['cases']);memory=max(out['peak_working_set_bytes'],out['peak_commit_bytes'])
    support={}
    for item in audit['ehu_records']:
        support.setdefault(item['room'],[0]*5)[item['label']]+=item['windows']
    supporttable='\n'.join('|'+room+'|'+'|'.join(map(str,counts))+'|' for room,counts in support.items())
    exclusions=Counter()
    for item in audit['ehu_records']:exclusions.update(item['excluded'])
    facts=dict(status='completed',feedback_score=.34,feedback_sha256='106ae34d405e651fc91230e9df17a44a338e8b0a00c4f88d5df86af19a4e8349',platform_best=.37,selected=selection,primary=primary,development_fits=len(dev),full_fits=len(full),pretraining_fits=len(pre),total_fits=len(fits)+len(pre),excluded_coarse_completed_fits=len(coarse_fits)+len(coarse_pre),mean_results=mean,cleaning_summary=audit['summary'],verification_status=verification['status'])
    save(WORK/'summary.json',facts)
    text=f'''# 0.34反馈后的数据修订与训练结果

用户报告上一版外部多任务seed42平台0.34，精确SHA256为106ae34d…a4e8349，比原0.37低0.03。只给这份模型及其同内容副本登记成绩，未给纯官方对照赋分，提交编号未知。已保存身份审计checks/external_upgrade_034_feedback.json。原两份0.37仍为平台已确认最佳。

本轮完成新数据下载、清洗、训练/测试分组和新训练：正式128频组版本{len(dev)}次官方房间开发拟合、{len(full)}次全48记录训练、{len(pre)}次预训练，共{len(fits)+len(pre)}次拟合。另有{len(coarse_fits)+len(coarse_pre)}次已完成的粗频组试验可恢复保留，不参与方案选择；中断的不算完整拟合。新导出文件平台尚未评测。

## 从0.34总结的经验

外部数据加入后并没有超过0.37；上一轮同时改变了新增路线、采样、辅助损失与训练数据，不能把下降完全归因于“外部数据有害”。共享逐频时间表示在自身对照中有帮助，但未证明能超越原结构。低训练损失、活动识别分数、二分类高分都不能代替老师五类跨房间计数。

因此本轮保留原0.37四路线拓扑，不加新残差分支；只预训练动态幅度卷积层，并比较预训练来源和微调强度。也不继续混用Brescia名义时间与SHARP活动标签，以避免重复引入已知的协议差异。

## 新数据与清洗

来源：[EHUNAM作者仓库](https://github.com/GuillermoDiazSM/EHUNAM-dataset)、[Figshare数据DOI](https://doi.org/10.6084/m9.figshare.28541225)、[Scientific Data2025论文](https://doi.org/10.1038/s41597-025-06238-4)，CC BY4.0。作者总包约77.73GB，本轮只按索引取需要的成员。

作者摘要只读分析后选择MC1、80MHz、0—4人PC/E、没有机器活动标签、真实Rx1、至少3000个CSI的记录，6个set属于5个物理房间。初选60条，一条教室4人记录实际时长不足，首尾裁剪后无窗；在未看模型分数的情况下，按同场景同人数未用候选的N_CSI补一条，合计选择61条，60条有可用窗。选中成员合计{verification['downloaded_selected_bytes']:,}字节，约{verification['downloaded_selected_bytes']/1e9:.2f}GB；另保留一份很小的20MHz格式检查样本，不用于训练。每成员CRC和SHA256验证，不宣称验证未下载的整个78GB包MD5。

实际MAT核对BW、N_People、People、Application、Rx与作者清单。空房E的N_People有时为空，只有Application明确为E且作者摘要标0时才标0；PC必须有明确人数且与参与者字母数一致。作者日期与原始文件名年份不一致时原样保留，不猜改。采集帧数不是有效时长，不能只按N_CSI挑数据。

按作者Subcarrier.m对MC1/80MHz去掉15个无效位置（241个有效频点），不套用SHARP的242频点规则。256原位置按相邻两点加权聚合为128频组，保留静默空组及频率轴，约0.625MHz/组，与原模型80MHz/128匹配。初始32粗分组会扩大局部卷积尺度，发现后归档重跑全部正式对照；没有用粗分组分数挑新模型。

采用实测秒时间戳，丢弃非有限/非递增帧；每条首尾各裁5秒以减少人员进出过渡，3秒无重叠切窗，覆盖不足2.8秒或存在>100ms断包窗口排除。按原中位采样率构造实测时间网格，再Kaiser低通多相重采样到约50Hz，最后取150点；不把插值说成真实新增观测。动态幅度只时间中心化和全窗归一化。排除计数：{dict(exclusions)}。

|EHUNAM物理房间|0人窗|1人窗|2人窗|3人窗|4人窗|
|---|---:|---:|---:|---:|---:|
{supporttable}

外部训练4个物理房间、5个set、50条记录，459窗，人数支持121/87/86/82/83。Office1整物理房间的10条记录、114窗全部留出，支持26/28/20/21/19，不是只覆盖0/1的伪五类测试。Basement两个set视为同一房间。多个独立RaspberryPi接收机不是相干天线，本轮只使用真实Rx1，不计算虚假的跨接收机相位。

RadioPoints混合对照保留原真实2Tx×3Rx、20MHz/30频点，以实测时间重采样，首尾各3秒不取；775个物理窗口的6条真实链路成为4650个单链路输入，仍只有15条原始记录。它们不是4650次独立实验，记录/房间不跨测试。20MHz/30约0.67MHz局部频率间隔接近80MHz/128，不扩成1024。

## 新训练/测试方式

保留官方原a1/c1/g1整房间留出以便比较，再增加e1整房间压力测试，专门看0/1和高人数误报。e1仅有0/1，分数不能作为完整五类准确率。所有官方未知房间记录不进入开发，最终全48条全部使用。150条U本轮不使用，避免把新的伪标签变量同时引入。

预训练单Rx二维残差CNN，四层64/128/256/384，频率无关的全局汇总加五类分类器，4×32更新，batch32。source_transfer仅用本折官方源记录的真实单Rx数据，匹配预算；ehu_transfer用新外部训练；ehu_radio交替EHUNAM/Radio；ehu_fullrate沿用同EHUNAM检查点，但微调更充分。Office1只作固定末轮评估，不用它早停、调参数或微调。

只迁移dynamic.layers卷积与归一化参数。首层1输入权重均分到原模型3个真实Rx输入属于权重初始化，不复制输入天线。其他三路、投影和输出层从相同种子正常初始化，最终参数仍19,706,024、原拓扑不变。前三变体首轮冻结动态卷积，其后LR4e-5，其他参数LR4e-4；fullrate全程4e-4不冻结。官方五轮，原loss、原类别/房间WeightedRandomSampler，训练与采样随机源分开。方案的完整记录、预测、权重、配置都保留。

## 实际结果

|方案|种子|原三房间均值|合并五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

选中外部方案{selection['variant']}，与source_transfer一起在2026初始化复验。两种子外部方案房间均值{mean[selection['variant']]['room_mean']:.1%}、宏召回{mean[selection['variant']]['macro']:.1%}；匹配官方源预训练对照{mean['source_transfer']['room_mean']:.1%}/{mean['source_transfer']['macro']:.1%}。原0.37结构历史两种子房间均值约27.0%，seed42约28.1%；与新训练不能直接当平台分数比较。fullrate和warm版本之间还改变了微调强度，不能把所有差异都算作新增数据效果。

两种子原三房间合并的逐类召回：

|真实人数|官方源预训练对照|新外部方案|
|---|---:|---:|
{recalltable}

这比只看均值更能说明哪里尚未解决。召回为0表示该开发类未被正确识别，不是没有测试样本；所有混淆矩阵保留真实支持数。没有据此反向更改验证标签或删除难样本。

|e1压力测试方案|0/1准确率|错误报2—4人的窗口|
|---|---:|---:|
{e1table}

原版新增e1压力测试为33.3%，60窗都预测1人，因此空房全错。e1不是新盲测集：此前数据参与过历史训练/分析，只是本轮每个开发模型严格整房间排除。整个本地开发数据已多轮使用，不承诺老师平台对应提升。

|仅预训练模型|种子|外部Office1五类准确率|
|---|---:|---:|
{external_table}

外部Office1的五类结果不等于老师分数，参与者可能跨房间重复，也不能称“所有人都未见过”。新PC数据以走动人数为主，不能认为补齐了静止多人场景；官方没有动作标签，不能凭错分断言某个房间的人静止或走动。相关推理只是待验证假设。

固定90%原版+10%纯外部二维专家探查：三房间均值{expert['room_mean']:.1%}，相对原版主分数仅{(expert['selection_score']-expert['base_selection_score'])*100:.2f}个百分点，满足门槛：{expert['selected']}。本轮没有按验证搜索融合权重，也未为不合格融合生成提交文件。

## 全量候选及核验

|方案|种子|训练数据上准确率|实际训练记录|不同窗口|
|---|---:|---:|---:|---:|
{fulltable}

该训练准确率不是独立测试。最终均覆盖全部48条，包括unknown三条，不能把训练高分当跨房间成功。

|文件|大小|本机最大用例均时|
|---|---:|---:|
{modeltable}

本轮外部研究候选副本：`submissions/data_revision_20261021/model.onnx`，对应`{primary['path']}`，SHA256 `{primary['sha256']}`，{primary['bytes']:,}字节。前一版0.34副本仍是`submissions/external_upgrade_20261021/model.onnx`，根目录model.onnx也没有替换。

本机真实/断包/零输入、ONNX18嵌入权重和float32接口均通过，最大用例平均{maxcpu:.3f}s、峰值约{memory/1e9:.2f}GB；只能证明本机资源合格，不能保证老师机器耗时或成绩。原始输入/训练缓存特征误差{verification['raw_cache_feature_max_error']:.3g}、预测误差{verification['raw_cache_logit_max_error']:.3g}；原49份模型及0.34模型哈希、成绩均保留。

## 结论与可复用经验

新增数据最先应解决真实标签、物理频率尺度、实测时间和记录隔离，不能把低训练损失当作清洗正确。Office1有完整0—4类支持，能暴露跨环境错误；e1压力测试能暴露空房/单人问题，不能被其他房间平均分掩盖。外部单Rx/多台接收机数据只能训练它实际提供的信号，不可硬造三天线相位。

新数据和结构迁移方案仍须与0.37及0.34反馈对照看待。是否值得提交应看整体与逐类表现，不因为某一折提高或训练达到70—90%就认定升级。已评测最高仍是0.37；这批新文件的平台分数留空，由用户实际提交后再登记。

后续优先方向：先围绕3/4人召回失败，核对逐记录动态分布和错误，而不是继续扩大参数；保持真实数据标签不变，比较有序人数损失与房间平衡的小规模单变量对照。若继续补数据，优先同一设备、同一房间内0—4人均覆盖且有静止/移动记录的采集；本轮仅增加走动人数数据没有解决这个缺口。这些是下一轮假设，尚未训练验证。

完整数据、旧粗分组归档、新权重与日志都在D盘data_revision_20261021。代码/计划在experiments/data_revision_20261021，逐文件与隔离核验见checks/data_revision_20261021.json。
'''
    (ROOT/'docs/DATA_REVISION_20261021.md').write_text(text,encoding='utf-8');(WORK/'REPORT.md').write_text(text,encoding='utf-8')
    registry=read(ROOT/'models/registry.json');assert registry['external_upgrade_20261021']['platform_score']==.34;registry['data_revision_20261021']=dict(platform_score=None,status='research_candidate_platform_unknown',primary=primary['path'],alias='submissions/data_revision_20261021/model.onnx',models=[{k:m[k] for k in ('path','bytes','sha256','platform_score')} for m in models],report='docs/DATA_REVISION_20261021.md',work_directory=str(WORK));save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:rd=csv.DictReader(f);fields=rd.fieldnames;existing=list(rd)
    existing=[r for r in existing if not r['path'].startswith('models/data_revision_20261021/')]
    existing.extend(dict(model='数据修订研究：'+m['variant']+' seed'+str(m['seed']),platform_score='',path=m['path'],sha256=m['sha256'],source='local training only',status='research_candidate_platform_unknown',submission_id='') for m in models)
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(existing)
    snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    print(json.dumps(dict(total_fits=len(fits)+len(pre),development=len(dev),full=len(full),pretraining=len(pre),means=mean,primary_path=primary['path'],size=primary['bytes'],old_best=.37,prior_score=.34),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
