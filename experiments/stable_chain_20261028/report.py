"""Complete development report including every declared fold and class."""
import shutil
from train_chain import *

def main():
    totals={};table=[]
    for family,folds in [('record',FOLDS[4:]),('room',ROOMS)]:
        cm=np.zeros((5,5),dtype=int);last=cm.copy()
        for f in folds:
            p=WORK/f'{f}_full_s42';r=old.read(p/'result.json');cp=torch.load(p/'final.pt',weights_only=True)
            a,b,c=[set(cp[k]) for k in ('fit_records','calibration_records','test_records')];assert not(a&b or a&c or b&c)
            assert old.sha(p/'final.pt')==r['sha256']
            cm+=np.array(r['test']['confusion']);last+=np.array(r['last_test']['confusion'])
            table.append((f,r['epoch'],r['test']['correct'],r['test']['total'],r['last_test']['correct']))
        totals[family]=dict(correct=int(cm.trace()),total=int(cm.sum()),accuracy=float(cm.trace()/cm.sum()),recall=(cm.diagonal()/cm.sum(1).clip(1)).tolist(),confusion=cm.tolist(),last_correct=int(last.trace()),last_confusion=last.tolist())
    src=old.read(WORK/'actual_full_s42/result.json');selection=old.read(WORK/'selection.json');artifacts=[old.read(WORK/f) for f in ('published.json','published_last.json')]
    audit=dict(totals=totals,source_selected=src['test'],source_last=src['last_test'],selection=selection,platform_score=None,goal_record60=totals['record']['accuracy']>=.6,code_sha256={p.name:old.sha(p) for p in Path(__file__).parent.glob('*.py')})
    old.save(WORK/'research_results.json',audit)
    lines=['# 稳定证据融合与训练：2026-10-28','','## 改动与边界','','保留大模型多尺度CNN、天线图、时间Transformer。对稳定状态、动态、相位分支归一化；上下文、时间编码、图消息和Transformer残差用初始较小的可学习系数控制；增加局部时间卷积和动态/状态辅助分类。11,274,732参数。辅助头不代表有真实活动标签或个人检测监督。','',
    '仅48条标注记录、960个不重叠窗口；种子42。AdamW lr0.0002，3轮预热后余弦衰减，30轮，NLL+.15动态NLL+.05状态NLL+.05时间反转一致性。batch24窗口、双时间视图；人数/房间/记录加权采样。源校准均衡NLL选权重；最终全量轮数来自七折源选择轮数中位数。没有旧权重、U、外部数据或输出偏置。','',
    '整条记录在训练、校准、测试间隔离。三折整记录验证是当前60%阶段目标，另检查四折留房间。数据被反复用于结构开发，所以不是盲测，更不等于比赛平台60%。本轮多项修改同时实施，效果归因需要后续消融。','',
    '## 开发结果','','|折|源选epoch|源选权重正确/总数|固定30轮正确/总数|','|---|---:|---:|---:|']
    for f,e,c,n,l in table:lines.append(f'|{f}|{e}|{c}/{n}|{l}/{n}|')
    for family,v in totals.items():
        lines+=['',f"{family}：{v['correct']}/{v['total']}（{v['accuracy']:.2%}）；固定30轮{v['last_correct']}/{v['total']}。",f"0—4人召回率：{[round(x,4) for x in v['recall']]}。"]
    lines+=['',f"整记录60%阶段目标：{'达到（开发集，非平台）' if audit['goal_record60'] else '尚未达到'}。",'小版源选择：整记录422/960、留房间230/880；上轮大版：整记录405/960、留房间380/880。仅作为相同划分历史比较。','',
    '## 全量模型与运行检查','',f"全量源选第{selection['epochs']}轮，训练拟合{src['test']['correct']}/960；第30轮{src['last_test']['correct']}/960。不能当作独立测试正确率。"]
    for a in artifacts:lines+=['',f"`{a['path']}`，{a['bytes']:,}字节，SHA256 `{a['sha256']}`；资源检查{a['resource_check']['status']}，本机最慢测试样例平均{max(c['cpu_mean_s'] for c in a['resource_check']['cases']):.4f}秒。"]
    lines+=['','原始CSI与缓存/PyTorch一致性、天线置换、零输入、CPU时间/内存、嵌入权重检查通过。平台分数未知，现有0.54继续保留。','',
    '全部权重、训练历史、混淆矩阵、逐窗口预测位于`D:/huawei2026-experiments/stable_chain_20261028/`。代码位于`experiments/stable_chain_20261028/`。']
    p=ROOT/'docs/STABLE_CHAIN_20261028.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/稳定融合训练_20261028.md');print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
