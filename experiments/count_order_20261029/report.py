"""Paired count/ranking evaluation without choosing epochs on target records."""
import sys,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
BASE=Path('D:/huawei2026-experiments');WORK=BASE/'count_order_20261029'

def evaluate(folder,checkpoint):
    d=torch.load(folder/'predictions.pt',weights_only=True);cp=torch.load(folder/('final.pt' if checkpoint=='selected' else 'last.pt'),weights_only=True)
    sets=[set(cp[k]) for k in ('fit_records','calibration_records','test_records')];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    ix=d['test_indices'];z=d['logits' if checkpoint=='selected' else 'last_logits'][ix];y=d['labels'][ix];expected=(z.exp()*torch.arange(5)).sum(-1);pred=z.argmax(-1);records=[]
    for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
        jj=[j for j,i in enumerate(ix) if d['rows'][i]['identifier']==ident];r=d['rows'][ix[jj[0]]]
        records.append(dict(identifier=ident,room=r['room'],count=int(y[jj[0]]),score=float(expected[jj].mean()),correct=int((pred[jj]==y[jj]).sum()),total=len(jj)))
    pairs=[(a,b) for a in records for b in records if a['room']==b['room']!='unknown' and a['count']>b['count']]
    return dict(correct=int((pred==y).sum()),total=len(ix),confusion=torch.bincount(y*5+pred,minlength=25).reshape(5,5).tolist(),mae=float((expected-y).abs().mean()),rank_correct=sum(a['score']>b['score'] for a,b in pairs),rank_total=len(pairs),records=records,epoch=cp['epoch'])

def main():
    results={};summary={}
    for name in ('texture_chain_20261028','count_order_20261029'):
      for cp in ('selected','last'):
        rr=[evaluate(BASE/name/f'record{i}_full_s42',cp) for i in range(3)];key=f'{name}_{cp}';results[key]=rr
        summary[key]=dict(correct=sum(r['correct'] for r in rr),total=sum(r['total'] for r in rr),rank_correct=sum(r['rank_correct'] for r in rr),rank_total=sum(r['rank_total'] for r in rr),mae=float(np.mean([r['mae'] for r in rr])),epochs=[r['epoch'] for r in rr],confusion=np.sum([r['confusion'] for r in rr],axis=0).tolist())
    rng=np.random.default_rng(29);cis={}
    for cp in ('selected','last'):
        base=results[f'texture_chain_20261028_{cp}'];new=results[f'count_order_20261029_{cp}'];groups=[]
        for a,b in zip(base,new):
            assert [r['identifier'] for r in a['records']]==[r['identifier'] for r in b['records']]
            groups.append(np.array([(y['correct']-x['correct'])/x['total'] for x,y in zip(a['records'],b['records'])]))
        draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]).mean() for _ in range(3000)];cis[cp]=np.percentile(draws,[2.5,97.5]).tolist()
    audit=dict(summary=summary,accuracy_delta_ci95=cis,results=results,platform_score=None,network_text_identical=(ROOT/'experiments/texture_chain_20261028/network.py').read_text()==(ROOT/'experiments/count_order_20261029/network.py').read_text())
    old.save(WORK/'summary.json',audit);tr=old.read(WORK/'trends.json')
    lines=['# 同房间人数顺序监督：2026-10-29','','本轮完成18个描述量的记录级趋势诊断，以及三折纹理网络对照训练，共60轮。只增加同房间、不同记录的人数顺序损失，结构、种子、数据、抽样与原纹理版相同；没有用测试标签调整特征或训练轮数。','',
    '## 为什么不直接使用运动强度数人','','a1与c1共同覆盖1—4人；a1仅11条、c1仅12条对应记录。运动能量与人数Spearman相关在a1约−0.886，在c1约+0.420；幅度高时间频率能量比例分别约+0.867、−0.402。简单的“人数更多→运动更强”不是两房间一致的规律。','',
    '这是小样本探索性相关，不能视作因果。18项描述量逐项检验存在多重比较，未用于挑选测试集上表现好的模型输入。缓存特征的秩不等于人数，时间频率组指窗口内周期数而非精确物理Hz。','',
    '## 单一改动','','原纹理结构543万参数不变。在每个训练batch中找同一已知房间、不同记录、人数不同的样本对。对较多人样本的预测期望人数，施加高于较少人的软约束：0.2×softplus(0.5×真实人数差−预测期望人数差)。它约束学习结果的顺序，不强迫某个原始能量指标单调。未知房间不拼成同一域。','',
    '配对只在训练记录中进行；推理不输入房间ID、不需要参考记录，也不增加模型头。其余仍为20轮、AdamW0.0003、batch24双视图、源校准均衡NLL选权重。房间匹配不能消除布局或采集批次混淆，仍需跨记录与跨房间验证。','',
    '## 对照结果','','|模型/权重|窗口正确/960|期望人数MAE|同房间记录对顺序正确/总数|源选轮数|','|---|---:|---:|---:|---|']
    for key,r in summary.items():lines.append(f"|{key}|{r['correct']}/960 ({r['correct']/960:.2%})|{r['mae']:.3f}|{r['rank_correct']}/{r['rank_total']}|{r['epochs']}|")
    lines+=['','排序分母是记录对，彼此相关，不能当作人数识别准确率，也不能把达到60%的排序率当作目标完成。模型选择只用源校准；两种训练阶段全部报告。','',f'按整记录、折内重采样3000次，新损失相对基线的窗口正确率差95%区间：{cis}。区间未涵盖重复结构探索和多随机种子不确定性。','',
    '这三个折是反复使用的开发集，不是隐藏平台。未进行本轮留房间或全数据训练，未导出新提交模型。0.6目标尚不能从训练拟合或排序任务宣布达到。','',
    '权重、预测、每轮配对数量与损失：`D:/huawei2026-experiments/count_order_20261029/`；代码：`experiments/count_order_20261029/`。']
    p=ROOT/'docs/COUNT_ORDER_20261029.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/同房间人数顺序实验_20261029.md');print(summary,cis,flush=True)
if __name__=='__main__':main()
