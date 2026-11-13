"""Publish only a checked candidate; preserve the scored054 model."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];W=Path('D:/huawei2026-experiments/ordinal_policy_20261107');E=W/'export';DEST=ROOT/'models/ordinal_head_20261107';NAME='01_gprime_ordinal75.onnx'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 resource=read(E/'resource_check.json');audit=read(E/'raw_audit.json');proof=read(E/'structure_proof.json');assert resource['status']==audit['status']=='pass';assert sha(E/'model.onnx')==resource['sha256']==proof['candidate_sha256'];registry=read(ROOT/'models/registry.json');best=registry['current_best'];assert sha(ROOT/best['path'])==proof['base_sha256']
 DEST.mkdir(parents=True,exist_ok=True);path=DEST/NAME
 if path.exists():assert sha(path)==resource['sha256']
 else:shutil.copy2(E/'model.onnx',path)
 meta=dict(id='ordinal75_20261107',name='Gprime054 ordinal75 direct25',path=path.relative_to(ROOT).as_posix(),sha256=resource['sha256'],bytes=resource['bytes'],platform_score=None,status='candidate_ready_for_platform_test',base_model=best['id'],base_platform_score=.54,change='Each view: .75 ordinal + .25 direct probabilities instead of .5/.5. All original weights, biases and policy node formulas preserved; altered scores can change policy branches.',training='Candidate reuses scored054 weights; six new objective-control fits and six linear probes did not yield a better replacement.',development=dict(room_proxy_baseline=278,room_proxy_candidate=284,total=880,corrected=12,broken=6,room_deltas=dict(a1=0,c1=5,g1=1,e1=0),record_bootstrap_difference95=[-.00454545444,.01931818244],context_record_control=477,context_record_ordinal75=485,context_total=960,actual_seen_records_baseline=676,actual_seen_records_candidate=679,actual_seen_total=960,seen_class_diagonal_baseline=[206,194,66,74,136],seen_class_diagonal_candidate=[204,203,61,76,135]),limitations='Small gain on reused development proxies, interval includes zero. Seen-record fit is not validation. Reduced seen-class2 accuracy. Hidden platform score unknown;0.60 not achieved.',resource=resource,raw_audit_cases=len(audit['cases']),report='docs/ORDINAL75_CANDIDATE_20261107.md')
 save(DEST/'model.json',meta);shutil.copy2(E/'resource_check.json',DEST/'resource_check.json');shutil.copy2(E/'structure_proof.json',DEST/'structure_proof.json')
 registry['ordinal75_20261107']=meta;registry['next_submission']=meta;registry['updated_date']='2026-11-07';registry['research_status']='ordinal75_candidate_checked_platform_unknown';assert registry['current_best']==best;save(ROOT/'models/registry.json',registry)
 ms=max(c['cpu_mean_s'] for c in resource['cases'])*1000
 text=f'''# 下一份候选：054内部有序人数分支占比提升

提交文件：`models/ordinal_head_20261107/{NAME}`。平台分数未知；已测最佳仍是原0.54。目标0.60尚未实现。

## 为什么这一轮给出候选

本轮先做6次从头训练（各5轮）：把原主要分类损失替换为最终融合NLL，得到430/960；进一步去掉专家与条件有序约束，得到457/960，均低于原上下文对照477/960。冻结上下文表示再拟合固定C=0.01的分类器，latent256为426/960，routes1024为404/960，也没有改善。没有把这些重训权重作为提交文件。

分支检查发现：原上下文权重单独用有序输出为487/960，原融合477/960；三个完整记录折均略好。有序占75%的保守组合为485/960（168、163、154；对照168、158、151）。因此转向检查原037结构的054规则下是否也有类似现象。

## 实际改动

原始每个视角的概率：0.5×直接分类 + 0.5×有序人数分布。

新候选每个视角的概率：0.25×直接分类 + 0.75×有序人数分布。

直接分类本身仍包含原分类器和四路线专家的混合；有序分布仍由原条件人数与有人/无人输出计算。没有新增人数偏置，没有新训练权重。12个视角、原Udirect偏置、0/2及1人保护公式、最终1人+0.35均保留。由于分数变了，部分样本选择的保护分支可能改变，不能称分支选择完全不变。

这是一项输出组合改进，不是新神经架构。原有序目标为已有学习信号，并未新增加静止/移动人物真值。

## 证据与代价

|留房间代理|原054流程|新组合|变化|
|---|---:|---:|---:|
|a1 /240|45|45|0|
|c1 /240|40|45|+5|
|g1 /340|174|175|+1|
|e1 /60|19|19|0|
|总计 /880|278|284|+6|

修正12个、破坏6个，净增0.68个百分点。按完整记录重采样的差值95%区间约[-0.45,+1.93]个百分点，包含零，尚不能证明稳定提高。这里是排除目标房间后重训的代理，不是原054实际隐藏测试成绩。纯有序组合为285/880，但a1下降2个，因此选保留25%直接分类的版本，不继续搜索细小比例。

原054真实全量权重在已见960窗口的缓存诊断：676→679。逐类正确数0/1/2/3/4人为[206,194,66,74,136]→[204,203,61,76,135]：1人增加9个、3人增加2个，但0人减少2个、2人减少5个、4人减少1个。这是训练记录诊断，不是泛化准确率，不能宣传成70.73%的测试成绩。

这些证据足以将其作为下一份小幅探索候选，但不足以预言平台超过054，更不能推算达到060。此次不修改平台排名。

## 数值与资源核验

- 大小：{resource['bytes']/1e6:.3f} MB，权重内嵌。
- 本机4线程CPU、各检查案例中最大平均推理耗时：{ms:.1f} ms。工具按现有项目限制100MB、0.7秒、6GB核验通过；老师机器耗时可能不同。
- {len(audit['cases'])}个原始输入案例核验，包括全部缓存中改变判断的窗口、每条记录首窗，以及零输入/正常/时间缺口案例。
- 保留原ONNX的单精度预处理；从原图提取直接/有序概率，再用独立NumPy公式计算期望输出，与新图最大差值{audit['max_error']:.3g}。
- 所有原初始化张量逐字节相同；仅替换12个视角末端的融合运算，新增两个比例常量。原054文件哈希复核未变。
- SHA256：`{resource['sha256']}`。

最初代理核对混用了较新的预处理缓存，触发数值不一致检查并中止；后改用原代理的相同缓存后复现通过。该失败运行没有作为准确率证据。提交图原始数据核验与缓存代理评估分开记录。

训练/核验代码：`experiments/final_objective_20261107/`。大型结果与检查点：`D:/huawei2026-experiments/final_objective_20261107/`、`context_readout_20261107/`、`ordinal_policy_20261107/`。
'''
 (ROOT/'docs/ORDINAL75_CANDIDATE_20261107.md').write_text(text,encoding='utf-8');p=ROOT/'README.md';old=p.read_text(encoding='utf-8-sig');prefix=f'最新待测（2026-11-07）：[01_gprime_ordinal75.onnx](models/ordinal_head_20261107/{NAME})，保留054权重与保护规则，把每视角内部有序人数分支占比从50%改为75%。留房间代理278→284/880，增益小且区间包含零；已见记录2人表现有损失。资源/原始数值核验通过，平台成绩未知，已测最佳仍0.54。见[说明](docs/ORDINAL75_CANDIDATE_20261107.md)。以下为历史。\n\n';p.write_text(prefix+old,encoding='utf-8');print(json.dumps(dict(path=str(path),sha256=meta['sha256'],bytes=meta['bytes'],cpu_max_mean_ms=ms),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
