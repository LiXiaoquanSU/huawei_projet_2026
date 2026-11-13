"""Publish one checked selective expert; scored054 remains authoritative."""
import csv,shutil
from n53_common import *
def main():
    folder=WORK/'export';s=read(WORK/'selection.json');v=read(folder/'validation.json');r=read(folder/'resource_check.json');proof=read(folder/'structure_proof.json')
    assert s['eligible'] and v['raw_windows']==960 and r['status']=='pass';assert sha(folder/'model.onnx')==r['sha256']==proof['candidate_sha256'];assert sha(SOURCE)==SOURCE_SHA
    dest=ROOT/'models/next053_20261025/01_g54_selective_lowcount.onnx';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.with_suffix('.json').exists():assert read(dest.with_suffix('.json'))['platform_score'] is None
    shutil.copy2(folder/'model.onnx',dest);assert sha(dest)==r['sha256']
    proxies=[x for x in read(WORK/'results.json') if x['room']!='actual' and x['method']==s['method']];correct=sum(x['metrics']['correct'] for x in proxies);total=sum(x['metrics']['total'] for x in proxies)
    meta=dict(id='selective_lowcount_20261025',name='0.54高人数保护与低人数选择性更新',path=dest.relative_to(ROOT).as_posix(),sha256=r['sha256'],bytes=dest.stat().st_size,
      platform_score=None,status='awaiting_platform',owner='self_selective_expert_on_teammate054',source='new checked selective experiment; platform unknown',report='docs/NEXT053_20261025.md',experiment_root=str(WORK),
      base_sha256=SOURCE_SHA,base_platform_score=.54,selection=s,local_validation=v,resource_check=r,independent_proxy_results=proxies,proxy_correct=correct,proxy_total=total,
      method='Exact original054preprocessing,batched12views. If054finalargmax>=2returnitsoriginalscores; otherwise usepreviousquarterencoderexpert. No newoutputbias.',
      training='No new weight fit this round; prior pairedseed42phase/timeencoder training reused; new evaluation cache extracted from actualONNX.',
      caveats='Source trained diagnostics; whole-room development reused. One proxy loses1window, source1person loses1. Not proven better than054 or reaching060.')
    save(dest.with_suffix('.json'),meta)
    for name in ('validation.json','resource_check.json','structure_proof.json'):shutil.copy2(folder/name,dest.parent/name)
    reg=read(ROOT/'models/registry.json');assert reg['current_best']['platform_score']==.54
    reg['research_candidates']=[x for x in reg.get('research_candidates',[]) if x.get('id')!=meta['id']]+[meta];reg['next_submission']=meta
    reg['research_status']='selective_lowcount_checked_awaiting_platform';reg['candidate_note']='Global quarter update scored053. New selective version preserves054final2/3/4scores; exactONNXpreprocessing evaluation, all960rawwindows audited. Source1person/proxyrooma1eachlose1. No claimed060.'
    reg['latest_research']=dict(report=meta['report'],status='one_checked_selective_candidate',best_scored=.54);reg['target_platform_score']=.60;reg['target_status']=dict(target=.60,status='research_in_progress',best_known=.54)
    save(ROOT/'models/registry.json',reg)
    cp=ROOT/'models/platform_scores.csv'
    with cp.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[x for x in rows if x['sha256']!=meta['sha256']];new={k:'' for k in fields};new.update(model=meta['name'],path=meta['path'],sha256=meta['sha256'],source=meta['source'],status=meta['status'],availability='retained',original_path=meta['path']);rows.append(new)
    with cp.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    snapshot=WORK/'source_snapshot';snapshot.mkdir(exist_ok=True);codes={}
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name);codes[p.name]=sha(p)
    save(ROOT/'checks/next053_20261025.json',dict(metadata=meta,code_sha256=codes,permutation_check=read(WORK/'permutation_check.json'),prototype=read(WORK/'prototype/prototype.json')))
    summary=read(WORK/'summary.json')+read(WORK/'ordinal_summary.json');lines='\n'.join(f"|{x['method']}|{x['source']['correct']}|{x['proxy_net']:+d}|{'是' if x['primary_proxy_independent'] else '否，Gcal迁移压力测试'}|{'通过' if x['eligible'] else '未通过'}|" for x in summary)
    classlines='\n'.join(f"|{k}|{v['baseline_confusion_matrix'][k][k]}|{v['confusion_matrix'][k][k]}|" for k in range(5))
    roomlines='\n'.join(f"|{x['room']}|{x['metrics']['correct']-(x['metrics']['corrected']-x['metrics']['broken'])}/{x['metrics']['total']}|{x['metrics']['correct']}/{x['metrics']['total']}|{x['metrics']['corrected']}|{x['metrics']['broken']}|" for x in proxies)
    runtime=max(c['cpu_mean_s'] for c in r['cases'])
    report=f'''# 0.53反馈后的新一轮：选择性专家（2026-10-25）

## 平台记录

用户确认`01_g54_encoder_quarter_seed42.onnx`平台0.53，已按SHA256 `ac51e11fb24bb514f74fcf76d0f69c694e3c7e6909ec1c6cd6ab2bb7469bce27`登记；最佳仍队友0.54，目标仍突破0.60。0.53说明上一轮来源总体提升和代理净增不够预测平台，不能继续把全局编码器更新视为已证实有效。

新文件`{meta['path']}`，{meta['bytes']/1e6:.2f}MB，SHA256 `{meta['sha256']}`。**未有平台成绩，不宣称超过0.54或达到0.60。**

## 本轮新思路

把“是否采用适配器”从所有窗口全局使用改为条件专家：先完整计算0.54，**其最终argmax为2、3或4时返回原5维分数；最终为0或1时才切换到四分之一更新后的编码器专家**。条件在原0.54最终+0.35之后判断。适配器仍可输出0–4，并非被限制只分0/1；这使原模型低估为1的某些真实2人窗有机会修复。

原0.54内部的单视角1保护、时间正反0/2保护、全六Rx排列×正反时间fallback及最终1人+0.35保留。适配专家也沿用同样规则，但用更新后的phase.project、temporal.blocks.2、temporal.project。没有再增加类别偏置、房间识别或新隐藏标签。

适配权重是上一轮真实标注同房间0/1配对训练的seed42、25%参数更新，旧0.53文件中的同一组更新，原FP16/FP32精度保持。本轮没有新拟合权重，不能写成重新训练了整个网络；旧训练使用的历史预处理尚未重训，本轮首先统一的是评估口径。

## 预处理口径已用于本轮评估

直接从实际已评分0.54图提取identity与时间反向的`cat_8`、`cat_6`预处理节点，重新计算48条标注记录的960窗，保存D盘t.npy/s.npy/rows.json。没有用近似的原生Torch预处理替代。S3接收通道特征排列与实际12视角预处理在8个原始窗口（含上轮失败第599窗）逐项比较，最大误差0。

这消除了本轮评估缓存与提交端数学口径不同的依赖。没有声称已完全解释上轮原生Torch/ONNX低方差数值差异，也没有修改原0.54预处理。全部教师/适配器均用这组同样的实际预处理特征作本轮推理。

## 计算结构的新改进

原图把12视角复制为12套顺序节点，虽然共享权重，预处理和图体仍有重复。本轮只做两次实际预处理（原时间、反时间），再按已验证的精确通道映射生成12份特征，合批进入一套原模型网络。选择性专家复用原静态/动态表示和相位/时间前缀，仅重算更新后缀和读出。

存储只压缩那些可被FP16精确表示、再转回FLOAT逐值不变的大数组，不引入额外量化误差。0.54等价原型41.89MB，三个普通/时间间隔/零输入探针与原文件logit差小于5e-7；粗测CPU约0.34–0.41秒/窗，该原型没有新平台分数。本轮实际候选还含更新后缀，最终{meta['bytes']/1e6:.2f}MB，不能把原型速度当最终速度。

## 实验与失败方案

本轮比较13种固定配置（含基线），每种在4个整房间教师和全标注教师上测试。没有扫偏置/阈值。

- 分离适配：全输出概率平均、只在原0/1区域接管、只重分配0/1总概率质量、只更新是否有人概率。
- 已评分意见：0.54和GNP0.53平均；再加入Gcal0.53的概率平均/逐类中位数/序数阈值logit融合。Gcal偏置表依赖全来源教师的U预测，故对应代理只能叫迁移压力测试，不作为严格独立房间证据。
- 频率新视角：在特征空间将128个幅度频率格、32个相位频率格反向，时间/summary保持；fallback分别混25%/50%。这不是已证明的物理不变性。25%源净增1但c1损1、g1增3；50%源净损3、c1损5，没有稳定收益，淘汰。此实验比较24视角，最终候选未使用频率镜像。
- 序数新融合：六Rx每个视角先计算P(人数≥1/2/3/4)，平均logit后重建合法有序分布，再混25%/50%原fallback。来源总分升，但4人分别136→127/115，代理无净增，淘汰。
- 只修0/1概率质量的版本来源纠正4、未改坏，代理仅g1增1，不作为本轮推荐；统一presence版本来源1人损失更多，也淘汰。

|配置|来源缓存正确/960|代理净增/880|可作独立教师对照|本轮条件|
|---|---:|---:|---|---|
{lines}

本轮条件是代理总增、至少2房间提高、无单房间下降超过2pp、来源4人正确不下降，并排除Gcal依赖代理。选择性接管是唯一通过者。没有声称所有类别都不退步：来源1人仍损1窗。历史房间反复用于开发，不能当全新盲测。各教师fit/calibration/test整记录分离，原教师和适配器fit均不含外层房间标签，保持各自原bias/temperature不重新校准。

## 实际960窗ONNX结果

本轮完整运行全部960个原始窗口，不再只检查148窗。候选 **{v['correct']}/960（{v['accuracy']:.2%}）**，同一批量实现中的0.54等价对照{v['base_correct']}/960。纠正{v['corrected']}、改坏{v['broken']}。这些窗口已参与原网络训练，仍是来源诊断，不能当作隐藏成绩。

|真实人数|0.54对照正确|新候选正确|
|---|---:|---:|
{classlines}

对原0.54最终预测2/3/4的所有来源窗口逐条验证：新候选的完整5维输出与同图对照逐值相等。其余区域允许新专家改判，因此0/1仍有风险；不能说改判全部正确。

## 整房间开发代理

合计{correct}/{total}（{correct/total:.2%}），原278/880（31.59%），净增11，即+1.25个百分点。两个房间提高，a1损1窗，e1持平。不能把32.84%对照当作平台0.54的同一尺度。

|外层房间|原正确/总窗|新正确/总窗|纠正|改坏|
|---|---:|---:|---:|---:|
{roomlines}

## 核对范围与资源

实际提交ONNX全部960原始输入，与独立PyTorch网络在相同实际预处理输出上的概率最大误差{v['independent_encoder_probability_max_error']:.3g}；边界argmax差异{v['borderline_reference_argmax_disagreements']}。预处理节点直接来自已评分图，并在提交图中逐字节保留；这里不声称独立原生预处理实现等价。

独立CPU四线程资源检查通过，各样例平均最慢{runtime:.3f}秒/窗，文件内嵌权重，接口仍frames[600,1,3,1024,2]、intervals[600]→5类FLOAT。原0.54文件SHA仍为`{SOURCE_SHA}`，未改动。D盘保留精确特征、各对照、逐条预测、程序快照；C盘只新增最终一个候选与小型记录。无GitHub推送。

## 接下来如何判断

平台反馈决定选择性接管是否适合隐藏分布。此处局部收益较小，不应预言0.60；如果没有迁移，下一步优先在本轮精确预处理缓存上重新训练专门的低估识别分支，并使用房间留出与高人数保持约束。避免因为来源总分更高就取消0.54高人数保护，也不要给失败的频率/序数方案补报成功。
'''
    (ROOT/meta['report']).write_text(report,encoding='utf-8')
    for filename,link in [('README.md','docs/NEXT053_20261025.md'),('docs/SUMMARY.md','NEXT053_20261025.md'),('models/README.md','../docs/NEXT053_20261025.md')]:
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1)
        p.write_text(heading+f'\n\n最新反馈：全局编码器更新版0.53，最佳仍0.54。新待测为选择性低人数更新`{meta["path"]}`（{meta["bytes"]/1e6:.2f}MB）：完整960原始ONNX{v["correct"]}/960、整房间开发代理{correct}/880；来源已训练，代理反复开发，均不是平台分数。已改用实际ONNX预处理构建本轮评估缓存，高人数输出保护已验证。[新实验]({link})。以下旧状态为历史记录。\n'+rest,encoding='utf-8')
    print('N53_PUBLISHED',dest,meta['sha256'],flush=True)
if __name__=='__main__':main()
