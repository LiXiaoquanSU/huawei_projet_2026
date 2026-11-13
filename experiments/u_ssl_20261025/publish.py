"""Publish exactly one checked encoder-finetuning candidate and its evidence."""
import csv, shutil
from ssl_common import *

def main():
    export=WORK/'export';v=read(export/'validation.json');r=read(export/'resource_check.json');p=read(export/'structure_proof.json');s=read(WORK/'selection.json')
    assert r['status']=='pass' and sha(export/'model.onnx')==r['sha256']==p['candidate_sha256']
    assert sha(ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx')==p['base_sha256']
    assert len(list((WORK/'fits').glob('*/complete.json')))==50
    dest=ROOT/'models/u_ssl_20261025/01_phase_time_finetune.onnx';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.with_suffix('.json').exists():assert read(dest.with_suffix('.json'))['platform_score'] is None
    shutil.copy2(export/'model.onnx',dest);assert sha(dest)==r['sha256']
    evaluations=read(WORK/'evaluations.json')
    proxies=[x for x in evaluations if x['room']!='actual' and all(x[k]==s[k] for k in ('method','amount','variant','deployment'))]
    total=sum(x['metrics']['total'] for x in proxies);correct=sum(x['metrics']['correct'] for x in proxies)
    meta=dict(id='u_ssl_20261025',name='相位与时间编码器微调版',path=dest.relative_to(ROOT).as_posix(),sha256=r['sha256'],bytes=dest.stat().st_size,
      platform_score=None,status='awaiting_platform',source='50 matched supervised/self-supervised/paired encoder fits; unscored candidate',
      report='docs/U_SSL_20261025.md',experiment_root=str(WORK),selection=s,local_validation=v,resource_check=r,independent_proxy_results=proxies,
      proxy_accuracy=correct/total,proxy_correct=correct,proxy_total=total,updated_encoder_parameters_per_seed=1512192,
      caveats='Source-trained diagnostics are not independent accuracy. Historical whole-room controls reused for development. No guarantee to improve platform049 or reach052.')
    save(dest.with_suffix('.json'),meta)
    for name in ('validation.json','resource_check.json','structure_proof.json'):shutil.copy2(export/name,dest.parent/name)
    registry=read(ROOT/'models/registry.json');assert registry['current_best']['platform_score']==.49
    registry['research_candidates']=[x for x in registry.get('research_candidates',[]) if x.get('id')!=meta['id']]+[meta]
    registry['next_submission']=meta;registry['research_status']='encoder_finetuning_checked_awaiting_platform';registry['candidate_note']='50 fits completed. Selected by room controls and source class preservation; U improvement not assumed.'
    registry['latest_research']=dict(report=meta['report'],status='checked_awaiting_platform',best_scored=.49);save(ROOT/'models/registry.json',registry)
    cp=ROOT/'models/platform_scores.csv'
    with cp.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[x for x in rows if x['sha256']!=meta['sha256']];new={k:'' for k in fields};new.update(model=meta['name'],path=meta['path'],sha256=meta['sha256'],source=meta['source'],status='awaiting_platform',availability='retained',original_path=meta['path']);rows.append(new)
    with cp.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    snapshot=WORK/'source_snapshot';snapshot.mkdir(exist_ok=True)
    code={}
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,snapshot/f.name);code[f.name]=sha(f)
    save(ROOT/'checks/u_ssl_20261025.json',dict(metadata=meta,code_sha256=code,raw_u_manifest_sha256=sha(WORK/'u_raw_manifest.json')))
    roomlines='\n'.join(f"|{x['room']}|{x['metrics']['correct']}/{x['metrics']['total']}|{x['metrics']['corrected']}|{x['metrics']['broken']}|" for x in proxies)
    classes='\n'.join(f"|{k}|{old}|{v['confusion_matrix'][k][k]}|" for k,old in enumerate([206,198,66,78,140]))
    comparison=[x for x in read(WORK/'summary.json') if x['amount']==.5 and x['variant']=='ensemble' and x['deployment']=='full']
    names={'supervised':'仅监督','source_ssl':'标注视图自监督','official_u':'官方U自监督','paired':'同房间0/1配对','paired_u':'配对＋官方U'}
    comparisonlines='\n'.join(f"|{names[x['method']]}|{x['source']['correct']}/960|{x['proxy_net']:+d}|{x['room_net']['g1']:+d}|" for x in comparison)
    worst=max(x['cpu_mean_s'] for x in r['cases'])
    report=f'''# 相位与时间编码器微调实验（2026-10-25）

待提交：`{meta['path']}`，{meta['bytes']/1e6:.2f}MB。SHA256 `{meta['sha256']}`。**尚未获得平台成绩；已评分最佳仍0.49，不能宣称达到0.52。**

## 最终选择

`{s['method']}`，参数插值{s['amount']}，{s['variant']}，{s['deployment']}。

{s.get('reason','')}

保留原0.49四视角、静态/动态早期编码器和全部读出头；实际更新phase.project、temporal.blocks.2、temporal.project，每种子1,512,192个编码器参数。原图与全部原初始化张量保持，附加更新后缀并复用原前缀。训练投影器不进入提交图。原类别偏置、温度及最终1人+0.35保持，没有重新扫偏置。

配对的出发点是：在同一个房间内比较空房与1人，房间背景相对接近，训练约束要求有人的窗口获得更高的1人相对分数，减少单靠房间线索判断的机会。它不等同自动去除房间信息，也没有额外真实动作标签；本轮结果仅支持作为受控训练候选。两个种子分别更新，最终逐视角平均概率，再进入保留的融合规则；每个种子先与原编码器参数各取一半，以限制微调漂移。

## 数据与对照

比赛标注48条、960个不重叠600帧窗；官方U150条，经逐文件形状/有限值/ATI检查后得到3000个窗。没有外部数据、U硬伪标签或假设U类别均匀。U按文件名组、记录、窗口抽样，文件名组不等同已证实的物理房间。

完成5方法×2种子×（4个整房间代理+全标注）=50次训练：监督、标注视图自监督、官方U视图自监督、同房间0/1配对、配对+U。每次240步，保留40/120/240；代理只按整记录calibration损失选快照，全数据使用代理所选步数中位数对应的快照。比较参数插值0.25/0.5/1及两种子融合，属于开发选择。

AdamW编码器lr5e-5、投影器lr5e-4、weight_decay0.01，源批量16，按类别/房间平衡；监督交叉熵含0.02标签平滑，另有0.1教师KL与0.02表示漂移约束。自监督额外32窗，VICReg权重0.02，比较同一窗口原视角与接收通道变换/时间反向，不能理解为不同房间必须同一表示。配对方法额外8对同房间0/1窗，二分类权重0.3、排序权重0.1，只使用真实存在的类别配对，因额外标签抽样不能把全部收益归于排序项。

初始30次显示U未稳定优于同预算监督/标注自监督；固定120步、0.5插值的全数据双种子3方法同为728/960。之后添加配对阶段20次。完整结果和固定预算对照在D盘summary.json、matched120.json；不能把加入U本身当作已证明的优势。

以下同为双种子、0.5插值，但快照由各自calibration选择，不能称完全相同步数：

|方法|来源正确数|整房间净增/880|g1净增|
|---|---:|---:|---:|
{comparisonlines}

固定120步配对与配对＋U的全数据双种子分别730/960和728/960，U在这组没有额外收益。固定严格0→1保护规则各方法没有触发，故未作为最终提交；本轮保留四视角内部路由，但允许更新后的编码器改变最终预测，而不是把原0/1结果强制锁死。

参考[VICReg原论文](https://arxiv.org/abs/2105.04906)的视图一致性、方差下限和协方差去冗余思想；不是复现论文成绩。

## 最终原始输入核对

960窗逐条运行实际提交ONNX：**{v['correct']}/960（{v['accuracy']:.2%}）**，原0.49为688/960；纠正{v['corrected']}、改坏{v['broken']}。这些数据已用于原始训练和本轮微调，**仅为训练来源诊断，不是独立测试准确率**。

|真实人数|原0.49正确窗|候选正确窗|
|---|---:|---:|
{classes}

主要收益集中2/3人边界；缓存诊断中2人正确66→93、3人78→87，1人仅198→201。1人误判空房仍93窗（原94），所以本轮没有解决最主要的漏人问题，不能根据总分上涨推断平台将涨3个百分点。

## 整房间代理

每个代理的原教师、微调、校准排除外层房间；原0.49风格的代理基线267/880（30.34%），候选{correct}/{total}（{correct/total:.2%}）。这是整房间分布迁移开发控制，与平台0.49不是同一统计口径。历史房间反复用于开发，不能称最终独立测试。

|留出房间|正确/总窗|纠正|改坏|
|---|---:|---:|---:|
{roomlines}

## 导出与限制

原始输入960窗与独立PyTorch后缀逐条比较，另检查零更新回归、零输入及时间间隔样例。独立CPU进程检查通过；4线程各样例平均推理最慢{worst:.3f}秒，文件内嵌权重，接口仍frames[600,1,3,1024,2]与intervals[600]，输出5类float32。平台机器速度、隐藏分布与下载链路仍需实际提交确认。

完整权重快照、U特征、原始预测、数据清单和对照留在`{WORK}`；C盘仅新增最终候选及小型报告，原两份0.49不改动。
'''
    (ROOT/meta['report']).write_text(report,encoding='utf-8')
    notice=f"本轮完成50次编码器微调训练，下一份待测为`{meta['path']}`（{meta['bytes']/1e6:.2f}MB）。源数据{v['correct']}/960，整房间开发代理{correct}/{total}；前者已参与训练，两者均不是平台分数。已评分最佳仍0.49。[实验报告]({{link}})。\n\n"
    for filename,link in [('README.md','docs/U_SSL_20261025.md'),('docs/SUMMARY.md','U_SSL_20261025.md'),('models/README.md','../docs/U_SSL_20261025.md')]:
        fp=ROOT/filename;text=fp.read_text(encoding='utf-8');heading,rest=text.split('\n',1)
        # Keep old notes as dated history, with the current status first.
        fp.write_text(heading+'\n\n'+notice.replace('{link}',link)+'以下为历史记录，待提交状态以以上说明和registry为准。\n'+rest,encoding='utf-8')
    print(json.dumps(dict(path=str(dest),sha256=meta['sha256'],validation=v,proxy_accuracy=correct/total,bytes=meta['bytes']),ensure_ascii=False))

if __name__=='__main__':main()
