"""Record actual training/validation results; do not promote unscored models."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from net import ROOT
from prepare import WORK,save
from run import read,sha
def main():
    verification=read(WORK/'verification.json');selection=read(WORK/'selection.json');screen=read(WORK/'screen_results.json');rep=read(WORK/'replicate_results.json');models=read(WORK/'export_registry.json');primary=read(WORK/'primary_candidate.json');prep=read(WORK/'external_preparation.json')
    fits=[read(p/'metrics.json') for p in sorted((WORK/'fits').iterdir()) if (p/'metrics.json').exists()]
    pretrains=[read(p/'metrics.json') for p in sorted((WORK/'pretrain').iterdir()) if (p/'metrics.json').exists()]
    stages=screen+[r for r in rep if r['seed']!=42]
    table='\n'.join(f"|{r['variant']}|{r['seed']}|{r['room_mean']:.1%}|{r['macro_recall']:.1%}|{r['worst_room']:.1%}|" for r in stages)
    means={v:dict(room_mean=float(np.mean([r['room_mean'] for r in rep if r['variant']==v])),macro=float(np.mean([r['macro_recall'] for r in rep if r['variant']==v]))) for v in ('source_control',selection['variant'])}
    delta=means[selection['variant']]['room_mean']-means['source_control']['room_mean']
    full=[m for m in fits if m['config']['fold']=='full'];ftable='\n'.join(f"|{m['config']['variant']}|{m['config']['seed']}|{m['training_evaluation']['accuracy']:.1%}|{m['distinct_records_seen']}|{m['distinct_windows_seen']}|" for m in full)
    resources=primary['resource'];cpu=max(r['cpu_mean_s'] for r in resources['cases']);memory=max(resources['peak_working_set_bytes'],resources['peak_commit_bytes'])
    modeltable='\n'.join(f"|{m['path']}|{m['bytes']/1e6:.2f} MB|{max(r['cpu_mean_s'] for r in m['resource']['cases']):.3f}s|" for m in models)
    ensemble=read(WORK/'ensemble_decision.json');anchor=read(WORK/'anchor_ensemble_decision.json')
    expert=read(WORK/'anchor_expert_decision.json')
    facts=dict(status='completed',development_fits=len(fits)-len(full),full_fits=len(full),pretraining_fits=len(pretrains),total_fits=len(fits)+len(pretrains),selection=selection,mean_comparison=means,external_vs_source_room_mean_delta=delta,primary=primary,platform_score=None,platform_best=.37,previous_models_verified=verification['old_model_hashes_verified'])
    save(WORK/'summary.json',facts)
    text=f'''# 外部数据升级训练结果（2026-10-21）

已完成{len(fits)-len(full)}次整房间开发训练、{len(full)}次全量训练、{len(pretrains)}次共享路线预训练，共{len(fits)+len(pretrains)}次拟合。新文件均为平台成绩未知的研究候选，不能把本地开发/训练准确率当成老师分数。原两份平台0.37仍为已确认最佳。

这轮外部数据已实际用于梯度训练，区别于上一轮仅清洗与随机森林诊断。

## 数据与结构

官方开发按a1/c1/g1整房间留出，所有记录窗口不跨折，未知房间三条不进入开发训练；全量阶段使用全部48条、960窗，每轮保证每条记录至少一窗。没有使用150条U，以单独评估外部有标签数据的作用。

外部RadioPoints：{prep['radio']['records']}条采集、{prep['radio']['shape'][0]}窗，真实2Tx×3Rx×30；Brescia：{prep['brescia']['records']}条采集、{prep['brescia']['shape'][0]}窗，仅原external_train，日期留出303窗不参与训练；SHARP：{prep['sharp']['records']}条采集、{prep['sharp']['shape'][0]}窗，空房/坐姿/步行/跑步活动标签。各源的名单、样本标签、时间来源在external_sequences.pt的rows与预训练清单中，外部多任务预训练实际覆盖15+27+13条记录。

保留原四路线（静态幅度、动态幅度、相位关系、时间编码）及存在/序数输出。新增共享时间CNN：每条真实链路/频点序列使用同一套卷积1→16→32→32，逐序列归一化，另保留相对变化强度；汇总各序列的均值/标准差/最大值形成256维表示，线性残差接到原融合层。输入链路数可以变化，不依赖天线排列，不复制天线或补造1024子载波。总训练参数19,842,961，比原版增加136,937。

RadioPoints/SHARP的3秒幅度序列在低通后重采样为150点；SHARP没有真实时间戳，485窗为名义6ms间隔。Brescia沿用v2的150点插值缓存并邻频聚合，未重做插值前抗混叠；其中120个训练窗时间为名义值。官方新路线从原缓存取得96×150序列，与ONNX现场预处理核对一致。它们不是校准后的多普勒，不能精确估算人体速度；坐姿标签也不等于完全静止。

预训练128更新；官方微调5轮AdamW4e-4。除原任务损失外，共享路线有人数辅助CE，外部批次继续更新防止遗忘；SHARP使用活动CE与存在BCE，不为官方样本编造动作标签。source_control使用同折官方源数据匹配额外预训练与辅助更新预算。每个开发模型均独立初始化，未加载见过留出房间的全量0.37权重。

## 开发结果

指标是三个完整留出房间的平均准确率、合并五类宏召回与最差房间，均不是平台成绩。

|方案|种子|房间均值|五类宏召回|最差房间|
|---|---:|---:|---:|---:|
{table}

external_counts只用外部人数；external_multitask再加SHARP活动；conditional_multitask再加同人数跨房间原型约束。首轮选中{selection['variant']}，再用2026初始化复验，并同时复验source_control。

两种子平均，外部方案房间均值{means[selection['variant']]['room_mean']:.1%}、五类宏召回{means[selection['variant']]['macro']:.1%}；同预算官方对照分别{means['source_control']['room_mean']:.1%}、{means['source_control']['macro']:.1%}，房间均值差{delta*100:+.2f}个百分点。原结构此前独立开发参考为seed42约28.1%、seed2026约25.9%；新增结构及采样并没有自动优于原结构。

新方案不能凭训练准确率或一个初始化的提升宣称稳定升级。房间数量仍很少、开发折已在多轮研究中使用，不能视为最终盲测。需同时检查每类混淆矩阵，尤其3/4人的召回；所有逐窗预测保留在fits目录。

## 固定融合对照

不平均不同初始化的参数，只试等权概率平均。同一外部方案两种子融合：房间均值{ensemble['room_mean']:.1%}、五类宏召回{ensemble['macro_recall']:.1%}、最差房间{ensemble['worst_room']:.1%}，达到预登记选择门槛：{ensemble['selected']}。

原版seed42锚点+外部单模型固定50/50融合：房间均值{anchor['room_mean']:.1%}、五类宏召回{anchor['macro_recall']:.1%}、最差房间{anchor['worst_room']:.1%}，达到预登记门槛：{anchor['selected']}。锚点开发分数只来自原版整房间留出预测，没有用已见过全部标签的0.37全量模型测试开发房间。最终选择方式：{selection.get('ensemble_kind','single')}。

后续探索只保留原版90%概率，加入10%纯外部专家（两种子仅外部预训练，未在官方标签上微调）：房间均值{expert['room_mean']:.1%}、宏召回{expert['macro_recall']:.1%}。相对原版仅提升{(expert['selection_score']-expert['anchor_selection_score'])*100:.2f}个百分点，未达预先规定0.5个百分点门槛，因此未导出此方案，也没有继续搜索融合比例。该结果不能当作已确认的新最佳。

如有双模型导出，权重FP16存储、Cast恢复FP32计算；通过和完整FP32参考的数值检查后使用。若资源或数值不通过则回退单模型，失败证据单独保留。

## 全量训练与文件

|方案|种子|训练数据评估准确率|实际覆盖记录|不同训练窗|
|---|---:|---:|---:|---:|
{ftable}

这一列只是已用于训练的数据上的评估，没有独立验证意义。全部48条记录每轮均进入训练，不是只训练a1/c1。全量训练不会再声称有独立本地测试成绩。

|候选文件|大小|本机CPU最大用例均时|
|---|---:|---:|
{modeltable}

本轮统一试提交副本：`submissions/external_upgrade_20261021/model.onnx`，来自`{primary['path']}`，{primary['bytes']:,}字节，SHA256 `{primary['sha256']}`。新模型平台分数尚未知，不能替换最佳登记。根目录旧model.onnx没有改写，避免把旧0.35文件误当新候选。

独立ONNX Runtime CPU进程检查真实、断包、零输入均通过：嵌入权重，输入float32[600,1,3,1024,2]与[600]、输出float32[5]；最大用例平均{cpu:.3f}秒、峰值内存约{memory/1e9:.2f}GB。只说明本机通过，不能保证老师机器的下载/运行与得分。

额外核查：原始输入与训练缓存最大特征误差{verification['cache_raw_feature_max_error']:.3g}、预测误差{verification['cache_raw_logits_max_error']:.3g}；共享时间编码器对链路排列的误差{verification['bank_channel_permutation_max_error']:.3g}；所有源/验证记录隔离，Brescia日期留出未训练；原49份ONNX哈希均未改变。

## 经验与限制

活动辅助训练提供了可迁移的运动表示，但外部数据的设备、采样、场景与运动方式仍有明显差异。仅增加数据量不足以补齐官方缺失的“房间×人数×运动状态”。强制同类原型接近这轮没有带来收益，可能压缩了同人数不同动作之间必要的表示差异；这是本轮结果的解释假设，不是已经证明的因果结论。

RadioPoints仅三个房间、每类每房间一条采集；Brescia3/4人仅少数日期/单天线，SHARP只有0/1。活动辅助任务的高分不等于多人计数高分。这一轮结果应与原版对照共同看待，不能承诺平台超过0.37或0.5。

来源/变换与作者归属随提交目录DATA_SOURCES.md保留。RadioPoints README写明非商业研究并要求引用；Brescia数据CC BY-SA4.0；SHARP数据页ODC Attribution、参考代码GPL另行保留。本轮只做本地研究训练与导出，未向平台上传或公开发布。

完整计划、代码、预训练权重、每折权重、历史、清单、预测与导出检查都在D:/huawei2026-experiments/external_upgrade_20261021；项目代码experiments/external_upgrade_20261021；审计checks/external_upgrade_20261021.json。
'''
    (ROOT/'docs/EXTERNAL_UPGRADE_20261021.md').write_text(text,encoding='utf-8');(WORK/'REPORT.md').write_text(text,encoding='utf-8')
    registry=read(ROOT/'models/registry.json');registry['external_upgrade_20261021']=dict(platform_score=None,status='research_candidates_platform_unknown',primary=primary['path'],submission_alias='submissions/external_upgrade_20261021/model.onnx',models=[{k:m[k] for k in ('path','sha256','bytes','platform_score')} for m in models],report='docs/EXTERNAL_UPGRADE_20261021.md',work_directory=str(WORK))
    save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;records=list(reader)
    records=[r for r in records if not r['path'].startswith('models/external_upgrade_20261021/')]
    records.extend(dict(model='外部数据升级研究：'+m['variant']+(' '+m['ensemble'] if 'ensemble' in m else ' seed'+str(m['seed'])),platform_score='',path=m['path'],sha256=m['sha256'],source='local external-data training; no platform evaluation',status='research_candidate_platform_unknown',submission_id='') for m in models)
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    snap=WORK/'source_snapshot'
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    print(json.dumps(facts,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
