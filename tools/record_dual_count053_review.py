"""Record the second053 and audit which historical changes actually raised platform scores."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'2026-10-25_037配方与种子复现'
REL='models/subnet_expectation_20261025/01_shared_backbone_dual_count_readout.onnx'
SHA='3b01b1406cfca9310ab7a22a3f065bd689f258f6c51624b811c148a4458b4437'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def write(p,fields,rr):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)

def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
    m=read(p.with_suffix('.json'));m.update(platform_score=.53,status='scored_below_best',source='user-reported platform0.53 for dualcountreadout;2026-10-25;submissionIDunknown')
    gains=[0]*5
    for item in m['selection']['rooms']:
        if item['room']=='actual':continue
        for k in range(5):gains[k]+=item['metrics']['confusion_matrix'][k][k]-item['baseline']['confusion_matrix'][k][k]
    assert gains==[0,8,25,-1,0] and sum(gains)==32
    m['feedback_analysis']=dict(local_per_class_net=gains,local_net=32,platform_delta_from_best=-.01,
       conclusion='Local net concentrated in class2, did not imply hidden-test improvement. No evidence hidden errors are the same records/classes.')
    save(p.with_suffix('.json'),m);save((PACKAGE/'新候选模型'/p.name).with_suffix('.json'),m)
    reg=read(ROOT/'models/registry.json');assert reg['current_best']['platform_score']==.54
    reg['research_candidates']=[m if x.get('sha256')==SHA else x for x in reg['research_candidates']]
    reg['latest_scored']=m
    if (reg.get('next_submission') or {}).get('sha256')==SHA:reg['next_submission']=None
    reg['research_status']='dualcount053_review_successful_directions';reg['candidate_note']='Two consecutive local-gain candidates scored053, below054. Review calibration/physicalTTA/conditionalretention gains; no new experiment requested in this turn.'
    save(ROOT/'models/registry.json',reg)
    f=ROOT/'models/platform_scores.csv';fields,rr=rows(f);found=[r for r in rr if r['sha256']==SHA];assert len(found)==1
    found[0].update(platform_score='0.53',status=m['status'],source=m['source']);write(f,fields,rr)
    f=PACKAGE/'platform_scores.csv';fields,rr=rows(f);r={k:'' for k in fields};r.update({'模型':m['name'],'平台分':'0.53','说明':'本地+3.64pp但平台低于054；净增32中2人类+25，非全类别改善','sha256':SHA,'文件路径':REL,'来源':m['source'],'记录状态':'已测低于最佳'})
    r[fields[0]]='10-25';rr=[x for x in rr if x.get('sha256')!=SHA]+[r];write(f,fields,rr)
    f=PACKAGE/'新增模型清单.csv';fields,rr=rows(f)
    for r in rr:
        if r['sha256']==SHA:r.update(平台分='0.53',状态='已测低于最佳054，不再待测')
    write(f,fields,rr)
    report='''# 平台提升复盘：哪些方向真正有效

已将共享主干双人数读出登记为平台0.53，精确SHA256为3b01b1406cfca9310ab7a22a3f065bd689f258f6c51624b811c148a4458b4437。它与上一份跨种子相邻向上版（也是0.53）是两个文件、两种方法。三份已测0.54保留，当前没有新的待提交首选。

## 真实上涨的高分主线

|平台变化|修改|经验|
|---|---|---|
|0.37→0.42|原权重不变，固定类别先验偏置|输出校准曾带来大收益；加权训练后为何仍有效不能仅凭成绩断言因果|
|0.42→0.44|换成U直接匹配偏置|利用目标侧无标签分布的校准有效；不等于原主干用U训练|
|0.44→0.45|原输入/时间反转两视角融合|标签不变的输入变换能释放已有权重信息|
|0.45→0.47|接收通道变换与保护，四视角|有效的是变换及保留规则的组合；同期十二视角也0.47|
|0.47→0.48|原始视角判1时优先保留|融合有时破坏可靠原判断，选择性保留有效|
|0.48→0.49|最终1人偏置+0.35|小幅决策边界修正有效；继续加到0.50反降0.48|
|0.49→0.54|保留保护规则，fallback改为完整6种Rx×2时间方向|大提升来自条件融合范围修正；不是重训主干或第一次使用十二视角|

以上是顺序接力的历史，不是各操作可独立叠加的因果收益。四/十二视角0.47与最后十二视角0.54并不矛盾：所保留的原判断及具体fallback逻辑不同。无保护直接十二视角平均曾只有0.41。

## 独立结构路线也曾进步，但不是054的底座

原四路线0.37；局部事件0.38；跨注意力/上下文各0.39，两者融合0.40。说明结构探索并非没有成果。但不能把0.40→0.42描述成在该融合模型上继续加偏置：已核验高分校准主线返回了原0.37权重。

高分主线0.44–0.54主要沿原权重做输出校准、物理变换视角融合、条件保留。复杂结构/更多参数/更多训练并没有在我们的已测记录中推动这条主线超过0.54；这不证明更好结构不存在。

## 当前为什么要调整优先级

1. 上轮跨种子相邻向上本地留房间+1.82pp，平台0.53；本轮双人数读出本地+3.64pp，平台仍0.53。重复出现本地收益不能迁移，需要审视开发评估的代表性，而不只继续改一层结构。
2. 本轮净增32个具体分布：0人0、1人+8、2人+25、3人−1、4人0。四个房间都改善不等于所有类别改善，主要修正的是本地2人误判。0和4净变化为0，也不表示其每条预测完全没变化。
3. 2–4人只在a1/c1出现，其他房间主要0/1；整房间隔离虽比随机分窗合理，仍不能代表隐藏房间的误差模式。相同开发房间反复筛选也会形成开发集适配。没有平台逐例标签，不能断定隐藏剩余错误也是本地这些错题。
4. 单次平台0.53对0.54的差距，不足以判断统计显著性；但结合多轮新训练/结构/辅助融合停滞，已足以调整试验优先级。

## 下一阶段应如何取舍（本轮只复盘，不启动新训练）

- 保留原054为控制，将校准和条件融合分开做单变量对照。输入变换优先满足人数标签不变、符合CSI物理/采集约定，不能任意造变换。
- 先查现有十二视角何时彼此冲突、哪些保护规则阻止有效证据；保持类别与原视角保护为控制，不能简单取消或盲目增加视角。
- 重新校准应以目标无标签数据诊断为依据，并独立检查每房间/每类别，而不是在旧标注来源上继续大规模搜索偏置。U没有标签，接近均匀不等于准确；更不能强迫每个房间均匀。
- 本地验证作为排除明显失败和资源问题的证据，不能继续把小幅代理提升当作平台候选排序的充分依据。新增数据首先解决独立房间覆盖和检验代表性，而非增加重叠窗口数量。

目标0.60尚未达到。该复盘没有承诺下一个模型能提升，也没有据此宣告结构方向无效。
'''
    (ROOT/'docs/SCORE_GAINS_REVIEW_20261025.md').write_text(report,encoding='utf-8')
    (PACKAGE/'平台提升方向复盘.md').write_text(report,encoding='utf-8')
    notice='\n\n最新反馈：共享主干双人数读出平台 **0.53**，与上一份跨种子候选同分，均低于最佳0.54。该文件不再待测；当前无新候选。见平台提升方向复盘（docs/SCORE_GAINS_REVIEW_20261025.md）。以下旧待测/交付状态为历史。\n'
    for rel in ['README.md','models/README.md','docs/SUBNET_EXPECTATION_20261025.md','docs/TEAM_PROGRESS_FULL_20261025.md']:
        f=ROOT/rel;text=f.read_text(encoding='utf-8')
        if notice.strip() not in text:heading,rest=text.split('\n',1);f.write_text(heading+notice+rest,encoding='utf-8')
    for src,dest in [('docs/SUBNET_EXPECTATION_20261025.md','后续实验_时间原点与子网络.md'),('docs/TEAM_PROGRESS_FULL_20261025.md','团队完整总结_含小全实验.md')]:
        (PACKAGE/dest).write_text((ROOT/src).read_text(encoding='utf-8'),encoding='utf-8')
    for f in [PACKAGE/'README.md',PACKAGE/'新候选模型/先读说明.md']:
        text=f.read_text(encoding='utf-8')
        if notice.strip() not in text:heading,rest=text.split('\n',1);f.write_text(heading+notice+rest,encoding='utf-8')
    print('Recorded second053; current best054 retained; next_submission cleared; local per-class net',gains)

if __name__=='__main__':main()
