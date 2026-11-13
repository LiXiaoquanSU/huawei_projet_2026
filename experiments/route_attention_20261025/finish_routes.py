"""Record negative cross-room results without publishing a worse model."""
import shutil
from ra_common import *

def main():
    assert sha(SOURCE)==SOURCE_SHA
    a=read(WORK/'results.json');b=read(WORK/'binary_results.json');assert not any(v['eligible'] for v in a+b)
    fits=list((WORK/'fits').glob('*/complete.json'));bfits=list((WORK/'binary_fits').glob('*/complete.json'));assert len(fits)==40 and len(bfits)==40
    d=torch.load(WORK/'actual.pt',weights_only=True);y=d['labels'];base=d['base'];cm=torch.bincount(y*5+base.argmax(-1),minlength=25).reshape(5,5)
    proxy_base={}
    for room in ('a1','c1','g1','e1'):
        p=torch.load(WORK/f'{room}.pt',weights_only=True);ix=old.split(room,p['rows'])[2];proxy_base[room]=old.metrics(p['base'],p['labels'],p['base'],ix)
        fit,cal,test=old.split(room,p['rows']);ids=lambda indexes:{p['rows'][i]['identifier'] for i in indexes}
        assert ids(fit).isdisjoint(ids(cal)) and ids(fit).isdisjoint(ids(test)) and ids(cal).isdisjoint(ids(test))
    summary=dict(status='completed_no_eligible_candidate',new_training_runs=80,distinct_families=8,source_base_correct=int(cm.trace()),source_total=960,proxy_base_correct=sum(x['correct'] for x in proxy_base.values()),proxy_total=880,route_best_proxy_net=max(x['proxy_net'] for x in a),binary_best_proxy_net=max(x['proxy_net'] for x in b),all_exported=False,new_platform_score=None,scored_best=.54,target=.60,target_achieved=False,source_confusion_matrix=cm.tolist(),proxy_baseline=proxy_base)
    save(WORK/'summary.json',summary)
    snapshot=WORK/'source_snapshot';snapshot.mkdir(exist_ok=True);codes={}
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name);codes[p.name]=sha(p)
    save(ROOT/'checks/route_attention_20261025.json',dict(summary=summary,source_sha256=SOURCE_SHA,code_sha256=codes,preprocessor_sha256=sha(PREV/'preprocessing.onnx'),initial_plan=read(WORK/'plan.json'),amended_plan=read(WORK/'binary_plan.json'),results=a,binary_results=b))
    def table(items):
        lines=[]
        for x in items:
            if len(x['seeds'])!=2 or x['amount']!=.5:continue
            r=x['rooms'];lines.append('|'+ '|'.join([x['architecture'],x['objective'],str(r['actual']['correct']),str(summary['proxy_base_correct']+x['proxy_net']),*[f"{r[k]['corrected']-r[k]['broken']:+d}" for k in ('a1','c1','g1','e1')]])+'|')
        return '\n'.join(lines)
    report=f'''# 平台0.54后的结构训练实验（2026-10-25）

用户确认选择性低人数版`models/next053_20261025/01_g54_selective_lowcount.onnx`为**平台0.54**。其SHA256为`{SOURCE_SHA}`，已登记，与队友Gprime0.54并列。两份原文件均保留。目标0.60尚未达到。

本轮完成**80次训练运行**：第一组40次、修改后的第二组40次。每组2结构×2目标函数×2种子×5教师划分；不是80个独立结构或80个独立测试集。线性分支两个种子会产生同样的确定性结果，不能拿它们当重复试验证据。**全部未通过开发筛选，本轮没有导出新的ONNX，也没有替换已评分模型。**

## 数据、精度与验证范围

48条比赛标注记录，960个不重叠600帧窗，不添加外部数据或U伪标签。用实际已评分原054图提取的预处理缓存；本轮新分支训练和评估都使用该精度口径。复用五个冻结教师（全数据及留出a1/c1/g1/e1），其相位/时间25%适配器沿用旧实验，本轮不重新训练主干。

每个代理按原manifest整条记录分离fit/calibration/test，新增分支不使用外层房间标签拟合。calibration选50/150/400步；全数据模型使用各代理所选步数的中位数就近取快照。代理已在历史实验反复开发，且此轮也据其结果继续改设计，**不能称独立盲测**。选择校正强度0.25/0.5/1也是开发选择。

基线全数据诊断679/960（70.73%），代理289/880（32.84%）。这两个数字均不是平台0.54的同一统计尺度。来源数据已参与原始训练和本轮训练。

## 第一组：三路交互残差

从12个接收排列×时间方向视角提取动态、相位、时间各256维表示；每路按视角求均值/标准差，得到3个512维输入。比较共享投影后简单池化与3路4头注意力。均值/标准差对视角顺序不敏感，但不因此保证去掉房间信息。未使用静态路或绝对summary，不声称三路对应真实静止/慢走/快走检测器。

冻结主干和原已评分规则，只训练新增小分支。最后层零初始化，tanh后减去类别均值；校正幅度受限，0.54最终预测2/3/4的窗口完整5维输出保持。原预测0/1的窗口可被新分支改到任何人数。训练为全类别shadow损失，并非只用0/1标签。

AdamW lr0.001、weight decay0.05、梯度裁剪1；类别/房间组均衡，或乘性更新困难组权重的正则化DRO变体。标签平滑0.03、教师KL0.1、残差L2 0.02、随机8/12视角一致性0.02。原预测低人数样本权重1，其余0.25。这是自定义目标，不是完整复现论文。

结构参考[Set Transformer](https://arxiv.org/abs/1810.00825)的集合输入思想；本实验用固定矩聚合后跨3路注意力，并非该论文架构复现。困难组训练参考[Group DRO正则化研究](https://arxiv.org/abs/1911.08731)，不把论文结果当本题性能保证。

下表统一取双种子、强度0.5。单种子和所有强度完整记录D盘results.json。

|结构|目标|来源正确/960|代理正确/880|a1净增|c1净增|g1净增|e1净增|
|---|---|---:|---:|---:|---:|---:|---:|
{table(a)}

注意力均衡版来源增加33窗（679→712），代理却净损94窗，g1损100窗。全配置最小代理损失仍为20窗。失败不能靠来源正确率较高掩盖，也不临时放宽“总净增、至少两房间提高、任何房间下降≤2pp”的筛选条件。

额外逐类检查发现：留出g1的某个注意力版本会把原预测1的窗口推成3/4，说明全类别校正存在明显过计数风险；不能只把原因归结为初始化。

## 第二组：相对视角的0/1专用分支

第一组失败后另存binary_plan.json再训练。以34维关系特征替代绝对表示：三路表示的视角余弦相似度统计、正反时间关系、路间相似度图差异、视角0/1相对log-odds及原模型0/1置信度。它消除了直接使用绝对表示均值/幅度的路径，但仍不能声称完全无房间信息。

比较线性分支与16维隐藏层MLP，只用训练侧真实0/1记录拟合二分类，输出有界反对称0/1修正。原2+预测保持；如果新修正反而让2/3/4成为最高分，则退回原输出。这样可以消除第一组引入的低人数→高人数升级路径，但仍会在0/1之间改错。

AdamW lr0.002、weight decay0.1、残差L2 0.05、标签软化0.03；训练侧特征均值/标准差用于标准化，未用外层房间求标准化参数。比较均衡组损失和DRO变体。其他划分、步数和筛选条件一致。

|结构|目标|来源正确/960|代理正确/880|a1净增|c1净增|g1净增|e1净增|
|---|---|---:|---:|---:|---:|---:|---:|
{table(b)}

所有单种子/双种子、三个强度均未通过。最小代理损失4窗；该线性版本来源689/960，但只有1个房间改善，因此仍淘汰。没有把“不如以前差”写成新进展。

## 当前错误的含义

当前054在来源数据的混淆矩阵（行真实、列预测0–4）：

```
{cm.numpy()}
```

最明显的来源问题是1人→空房93窗、2人→3人47窗、3人→4人45窗。**但留出g1的基线错误恰好不同：140个空房中127个被判1，200个1人中181个正确。** 因此简单学习“提高1人的分数”可能补来源漏人，却加剧新房间误报。不能从来源错题直接推断老师隐藏集也有同样错误分布。

本轮证据表明：新增注意力和困难组目标能够拟合来源错误，但没有显示可迁移修正；改成关系特征和二分类限制只减少了损害。它不证明更复杂结构永远无效，也不能证明054已到上限。

## 后续优先级

优先检查“哪些错误方向可跨记录/房间重复”，而不是继续扩大残差头或扫类别偏置。下一轮值得设计记录级交叉拟合的修正可信度/拒绝机制：修正分支需在完全未参与其训练的记录上证明收益，不在支持不足的信号分布上强行接管；与不拒绝的同结构做对照。先明确拒绝条件和固定比较预算，再看外层结果，不能看完g1损失后专门为g1加规则。该机制尚未实施，不把它计入本轮已完成结果。

U视图自监督此前没有稳定额外收益；本轮不声称“继续加U就能到0.60”。同理，没有隐藏标签支持房间识别或推断隐藏集人数先验。

## 保存与提交状态

全部训练快照、逐方法预测、精确特征、计划和程序快照保留`D:/huawei2026-experiments/route_attention_20261025`。本地检查摘要在`checks/route_attention_20261025.json`。本轮未导出模型，故不声称通过ONNX资源测试。export_routes.py/audit_routes.py仅是未执行的导出准备工具。

建议继续保留实际已验证的两份0.54，不为这组失败实验消耗平台提交次数。没有Git推送，没有删除已有文件。当前最好平台分数仍0.54，目标0.60未达成。
'''
    (ROOT/'docs/ROUTE_ATTENTION_20261025.md').write_text(report,encoding='utf-8')
    reg=read(ROOT/'models/registry.json');assert reg['best_own']['sha256']==SOURCE_SHA and reg['best_own']['platform_score']==.54
    reg['research_status']='route_and_relative_binary_experiments_completed_no_candidate';reg['next_submission']=None
    reg['latest_research']=dict(report='docs/ROUTE_ATTENTION_20261025.md',status=summary['status'],experiment_root=str(WORK),training_runs=80,best_scored=.54,new_model_exported=False)
    reg['candidate_note']='Both scored054 retained. Eight residual/specialist families,80fits, allfailed fixed cross-room criterion. No new submission; source gains didnottransfer. Target060 remains unmet.'
    reg['target_status']=dict(target=.60,status='research_in_progress',best_known=.54);save(ROOT/'models/registry.json',reg)
    for filename,link in [('README.md','docs/ROUTE_ATTENTION_20261025.md'),('docs/SUMMARY.md','ROUTE_ATTENTION_20261025.md'),('models/README.md','../docs/ROUTE_ATTENTION_20261025.md')]:
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1);p.write_text(heading+f'\n\n最新：选择性低人数版平台0.54，与队友Gprime并列。本轮完成80次结构/训练运行，均未通过跨房间开发筛选，未导出新模型；已评分054保留，目标060未达到。[完整记录]({link})。以下旧待测状态为历史。\n'+rest,encoding='utf-8')
    print('RA_FINISHED',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
