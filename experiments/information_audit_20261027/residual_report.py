import json,shutil
import numpy as np
import torch
from residual import ROOT,WORK
def main():
    r=json.loads((WORK/'results.json').read_text());agg=json.loads((WORK/'aggregate.json').read_text());assert len(r)==40
    margins={}
    for room in ('a1','c1','g1','e1'):
        d=torch.load(WORK/f'{room}_both_s42.pt',weights_only=True);ix=d['test_indices'];z=d['baseline'][ix];y=d['labels'][ix];top=z.topk(2,-1).values;gap=top[:,0]-top[:,1];correct=z.argmax(-1)==y
        margins[room]=dict(correct=int(correct.sum()),correct_margin_le07=int(((gap<=.7)&correct).sum()),median_correct_margin=float(gap[correct].median()),total=len(ix))
    for name,info in r.items():
        d=torch.load(WORK/f'{name}.pt',weights_only=True);ix=d['test_indices'];delta=d['logits']-d['baseline'];assert float((delta.amax(-1)-delta.amin(-1)).max())<=.70001
        assert info['nll']<=info['baseline_nll']+1e-6
        if info['epoch']:
            selected=next(h for h in info['history'] if h['epoch']==info['epoch'] and h['alpha']==info['alpha']);assert selected['calibration']['corrected']>=selected['calibration']['broken']
        z=d['baseline'][ix];gap=z.topk(2,-1).values;safe=gap[:,0]-gap[:,1]>.70001;assert torch.equal(z.argmax(-1)[safe],d['logits'][ix].argmax(-1)[safe])
    lines=['# 冻结骨干＋受限细节修正：2026-10-27','','完成40组小分支训练（最初32组，随后补8组同参数量控制）。没有更新原骨干，没有新平台成绩；最佳仍0.54。','',
    '## 方法','','四个留房间代理；每个源训练记录的基础预测来自未见过该记录的内层教师。复用已有12个内层教师，没有新增全主干训练。内层与外层训练、校准均排除目标房间。','',
    '分支为输入→32维SiLU→5类，最后一层零初始化。输出先限制到±0.35再去均值，任意两类别的得分差最多改变0.7。每组训练60步全批次AdamW；人数×房间均衡NLL，加对原本正确源样本的KL约束。仅源训练拟合特征标准化，标准化值裁剪到±8。','',
    '每5步检查源校准损失；修正系数0/0.25/0.5/1中选择。要求源校准改正数不少于误伤数，原模型不修正始终是候选。所有目标房间指标在训练/选择后计算。内层教师偏置已经使用同一套源校准标签，残差选择再次使用它；所以源校准分数不能视为独立验证，外层房间仍完全未参与训练/选择。','',
    '## 结果','','基线：278/880（31.59%）。以下均为本地重训代理，不是实际平台054模型。','',
    '|输入|种子|正确/880|改正|误伤|净变化|','|---|---:|---:|---:|---:|---:|']
    for name,v in agg.items():lines.append(f"|{name.rsplit('_s',1)[0]}|{name.rsplit('_s',1)[1]}|{v['correct']}/880|{v['corrected']}|{v['broken']}|{v['correct']-278:+d}|")
    lines+=['','summary只用原有摘要；time/phase分别增加时间/相位细节；both同时增加。capacity_control与both均为333维输入、同初始化和同参数量，但320个细节输入全部置零，因此有效输入维数不同。这个控制是在第一批结果之后补充，不是预注册对照。','',
    '## 为什么小修正也能造成很多误伤？','','|留出房间|基线正确数|其中第一、第二名得分差≤0.7|正确样本得分差中位数|','|---|---:|---:|---:|']
    for room,v in margins.items():lines.append(f"|{room}|{v['correct']}|{v['correct_margin_le07']}|{v['median_correct_margin']:.4f}|")
    lines+=['','得分差是模型输出的对数得分差，不是准确率，也不保证是校准置信度。≤0.7表示可能被这个幅度的修正翻转，并非一定翻转。已核验所有40组：超过上界的样本预测没有变化。限制输出幅度只能保护大间隔判断，不能保护原本正确但低间隔的判断。','',
    '## 结论','','这批修正均未超过基线，因此不导出或推荐提交模型，也不继续扫描修正幅度以挑选局部最好结果。原有摘要控制也会退化，说明问题不能全部归咎于新细节；源校准选中的修正方向未能可靠迁移到目标房间。','',
    '下一阶段应优先检查修正学习的验证方式：源记录交叉拟合仍可能包含同房间记录，不能保证学到跨房间纠错方向；同时基础偏置和修正分支复用校准集，可能使选择过于乐观。可检验按源房间交叉拟合的修正训练、独立的偏置/修正校准划分，以及保留为零修正的明确基线。但高人数只出现在a1/c1，分开房间后监督覆盖不足，不能承诺这个更严格协议一定可行或提高。','',
    '本轮只做了留房间残差对照，没有把全数据054骨干用于整记录验证；前一轮简单特征分类器的整记录40.42%不是本轮新分支的成绩。','',
    '## 产物','','`D:/huawei2026-experiments/detail_residual_20261027/`保存40份权重/预测、源校准扫描记录和汇总。代码为`experiments/information_audit_20261027/residual.py`与`residual_report.py`。原平台模型未修改。']
    path=ROOT/'docs/DETAIL_RESIDUAL_20261027.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(path,ROOT/'2026-10-25_037配方与种子复现/受限细节修正对照_20261027.md')
    regpath=ROOT/'models/registry.json';reg=json.loads(regpath.read_text(encoding='utf-8-sig'));assert reg['next_submission'] is None;reg.update(research_status='bounded_detail_residual_failed',research_report='docs/DETAIL_RESIDUAL_20261027.md');reg['bounded_detail_residual_20261027']=dict(fits=40,aggregate=agg,margins=margins,platform_score=None);regpath.write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(aggregate=agg,margins=margins)),flush=True)
if __name__=='__main__':main()
