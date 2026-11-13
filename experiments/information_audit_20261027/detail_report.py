import json,shutil,hashlib
from pathlib import Path
import numpy as np
from detail_features import ROOT,WORK
def main():
    r=json.loads((WORK/'probe_results.json').read_text());rng=np.random.default_rng(27);intervals={}
    for group,folds in [('room',('a1','c1','g1','e1')),('record',('record0','record1','record2'))]:
        for variant in ('summary_time','summary_phase','summary_both'):
            ds=[]
            for f in folds:
                a=r['folds'][f]['variants'][variant]['records'];b=r['folds'][f]['variants']['summary']['records'];assert [v['identifier'] for v in a]==[v['identifier'] for v in b]
                ds.append(np.array([[aa['correct']-bb['correct'],aa['total']] for aa,bb in zip(a,b)]))
            values=[]
            for _ in range(5000):
                summed=sum((d[rng.integers(len(d),size=len(d))].sum(0) for d in ds));values.append(float(summed[0]/summed[1]))
            intervals[group+'_'+variant]=np.percentile(values,[2.5,97.5]).tolist()
    names={'summary':'已有强度/时间摘要','summary_time':'摘要＋时间细节','summary_phase':'摘要＋相位相干度','summary_both':'摘要＋两类细节'}
    lines=['# 时间细节与相位相干度对照：2026-10-27','','本轮从48条标注记录提取全部960个非重叠窗口的新特征，完成84次逻辑回归拟合。没有训练新神经主干，没有导出提交模型。平台最佳仍0.54。','',
    '## 特征假设','','时间分支保留逐帧变化与4帧内残差的相对能量：600帧、3接收通道，先按32子载波池化，再统计每频组；相位分支保留每对天线的复相位平均模长、瞬时相干度平均和时间离散程度。三个接收通道/天线对用均值与标准差对称池化。时间特征128维，相位192维；基础摘要13维。','',
    '相位特征不保留绝对相位角，合成数据已验证对固定通道相位偏移、通道排列、时间反转的不变性，以及零输入数值有限。该检查不证明真实环境下鲁棒，也不能把相干度直接解释成人数。时间特征使用帧索引，不是精确非均匀采样时间重建。','',
    '## 验证与选择','','四个旧留房间开发折，训练和校准排除未知房间；另建三个按完整记录分组、人数分层的折，覆盖全部48条记录。后一种方式未保证所有房间都在训练中，尤其单记录房间可能整体落入测试，因此应称整记录留出，不称严格同房间验证。测试记录绝不进入训练/校准。每个源集合留出每类一整条记录校准；标准化只拟合源训练。','',
    '四组使用完全相同划分、加权采样等价的类别×房间样本权重和逻辑回归；C取0.01、0.1、1，仅按源校准类别均衡交叉熵选择。共7折×4组×3档=84次拟合。各组特征维数不同，结果不是严格相同容量神经网络消融。原平台权重未参与分类器训练和整记录验证。','',
    '## 结果','','|输入|留房间正确/880|准确率|整记录正确/960|准确率|','|---|---:|---:|---:|---:|']
    for v,name in names.items():
        a,b=r['room'][v],r['record'][v];lines.append(f"|{name}|{a['correct']}/{a['total']}|{a['correct']/a['total']:.2%}|{b['correct']}/{b['total']}|{b['correct']/b['total']:.2%}|")
    lines+=['','这里的基础摘要分类器不是旧054代理，也不是实际054模型；不能拿这个表直接预测平台分数。','',
    '|新增特征|留房间净增区间|整记录净增区间|','|---|---|---|']
    for v in ('summary_time','summary_phase','summary_both'):lines.append(f"|{names[v]}|{intervals['room_'+v]}|{intervals['record_'+v]}|")
    lines+=['','区间为按完整记录重采样、保持折分层的5000次bootstrap，数值为准确率差。没有校正多方案比较、折间训练集重叠和历史反复使用数据产生的不确定性，不是正式独立检验。','',
    '## 各折结果','','|折|摘要|＋时间|＋相位|＋两者|','|---|---:|---:|---:|---:|']
    for f,q in r['folds'].items():lines.append('|'+f+'|'+'|'.join(str(q['variants'][v]['correct'])+'/'+str(q['variants'][v]['total']) for v in names)+'|')
    lines+=['','## 本轮结论与后续','','本次两类细节共同加入后，整记录267→388/960（27.81%→40.42%），但整房间198→61/880（22.50%→6.93%）。同一种简单分类器在两个协议下给出相反结论，且整记录改善集中在record0和record2，record1没有净提升。它符合环境依赖线索在新房间失效的解释，但也可能涉及正则、样本覆盖和模型容量差异，不能凭这一项实验断定物理原因。',
    '', '不导出这套特征分类器供平台提交，也不据此断言细节无用。后续应检验：固定原骨干，新增零初始化、受限幅度的细节读出；同时保留仅原骨干及同容量对照，使用两套验证协议。训练时只用源记录选择权重，优先检查跨房间损失是否可以避免，而不是只优化整记录分数。原054全数据权重不能用于冒充留出训练的骨干。',
    '', '两套验证的排序冲突是当前需要解决的问题。管理员说有标签与测试同分布，不足以确定哪种协议单独代表平台；应结合官方房间划分说明和历史平台表现判断，不应根据本轮哪个分数好就挑哪个。','',
    '## 文件','','`D:/huawei2026-experiments/detail_features_20261027/`保存特征、逐折记录清单、28份预测/分类器参数及正则选择日志。代码位于`experiments/information_audit_20261027/detail_*.py`。本报告同时复制到队友资料目录。']
    path=ROOT/'docs/TIME_PHASE_DETAIL_20261027.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(path,ROOT/'2026-10-25_037配方与种子复现/时间与相位细节对照_20261027.md')
    regpath=ROOT/'models/registry.json';reg=json.loads(regpath.read_text(encoding='utf-8-sig'));reg.update(updated_date='2026-10-27',research_status='time_phase_detail_probes_completed',research_report='docs/TIME_PHASE_DETAIL_20261027.md');reg['detail_probes_20261027']={k:r[k] for k in ('room','record')};regpath.write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(results={k:r[k] for k in ('room','record')},intervals=intervals)),flush=True)
if __name__=='__main__':main()
