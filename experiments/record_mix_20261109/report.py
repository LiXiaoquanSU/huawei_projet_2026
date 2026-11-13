import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];D=Path('D:/huawei2026-experiments')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 rows=read(D/'next053_20261025/rows.json');out={}
 for folder in ['record_mix_20261109','conditional_routes_20261109','conditional_energy_20261109']:
  rr=read(D/folder/'results.json')
  for mode in sorted({r['mode'] for r in rr}):
   a=[r for r in rr if r['mode']==mode];assert len(a)==3
   cm=np.array([r['metrics']['confusion'] for r in a]).sum(0)
   out[mode]=dict(correct=int(cm.trace()),total=int(cm.sum()),fold_correct=[r['metrics']['correct'] for r in a],source_correct=sum(r['source_fit']['correct'] for r in a),confusion=cm.tolist(),class_recall=(cm.diagonal()/cm.sum(1)).tolist())
 contrasts={}
 for new,old in [('static_dynamic_energy','static_dynamic'),('static_dynamic','dynamic_dynamic'),('cross_record','same_record')]:
  def directory(mode):return 'conditional_energy_20261109' if mode.endswith('energy') else ('conditional_routes_20261109' if 'dynamic' in mode else 'record_mix_20261109')
  diffs=[];changed=fix=breaks=0
  for f in range(3):
   p=[torch.load(D/directory(m)/f'record{f}_{m}'/'predictions.pt',map_location='cpu',weights_only=False) for m in [new,old]]
   assert p[0]['indices']==p[1]['indices'] and torch.equal(p[0]['labels'],p[1]['labels'])
   a,b=[q['logits'].argmax(-1) for q in p];y=p[0]['labels'];changed+=int((a!=b).sum());fix+=int(((a==y)&(b!=y)).sum());breaks+=int(((a!=y)&(b==y)).sum());delta=((a==y).float()-(b==y).float()).numpy();ids=[rows[i]['identifier'] for i in p[0]['indices']];diffs.append(np.array([delta[np.array(ids)==k].mean() for k in sorted(set(ids))]))
  rng=np.random.default_rng(20261109);samples=[np.concatenate([rng.choice(d,len(d),replace=True) for d in diffs]).mean() for _ in range(10000)]
  contrasts[new+' - '+old]=dict(changed=changed,corrected=fix,broken=breaks,record_bootstrap95=np.quantile(samples,[.025,.975]).tolist())
 result=dict(date='2026-11-09',fits=15,epochs=180,aggregate=out,contrasts=contrasts,platform_best=.54,platform_score=None,next_submission=None,status='complete_no_export',limitations='Repeated development folds, not blind validation; bootstrap does not correct researcher selection.')
 save(Path(__file__).with_name('results_summary.json'),result)
 lines=['# 记录混合、条件分类与强度恢复实验（2026-11-09）','','完成15次训练、180轮。沿用按完整记录隔离的三折，每折训练640、验证320窗口。固定seed42、12轮、AdamW学习率0.0003；不依据外层成绩挑选轮数。全部是反复复用的开发集，不是老师隐藏测试。','','|方案|各折正确数/320|总正确/960|准确率|训练拟合/1920|','|---|---|---:|---:|---:|']
 for m,v in out.items():lines.append(f"|{m}|{v['fold_correct']}|{v['correct']}|{v['correct']/960:.2%}|{v['source_correct']}|")
 lines+=['','## 实验含义','','same_record与cross_record：同人数的潜在特征混合，对照分别取同一原始记录与不同原始记录。配对仅来自训练记录。两者计算量一致；此前无混合动态模型为389/960。跨记录混合未显示明显收益。','','dynamic_dynamic：两条动态编码路线，低人数头预测0/1/至少2；高人数头预测条件人数2/3/4。static_dynamic保持参数量一致，只把高人数路线输入替换为静态均值。最终五类概率通过条件概率乘法得到，使用五类负对数似然。它们不是有语义监督的静止人/运动人检测器。','','static_dynamic_energy：保留上述结构，恢复三对天线的去均值RMS强度，转换为log1p(1000*RMS)后送入低人数路线的摘要输入，另外14项置零，高人数路线摘要仍为零；不增加参数量、不拟合新偏置。强度可能来自活动或设备/环境变化，不能等同人数。','','## 成对错误变化与局限','']
 for k,v in contrasts.items():lines.append(f"- {k}：纠正{v['corrected']}个，破坏{v['broken']}个；按整记录、折内重采样的描述性95%差值区间{v['record_bootstrap95']}。")
 lines+=['','该区间没有校正多轮开发和方案选择偏差。窗口数不代表独立采集次数。条件分工的依据来自此前同一批开发数据的混淆矩阵，不能称为盲测。','','未达到本地60%，也没有证据说明能超过平台0.54。因此本轮不导出或推荐新提交模型；保留旧0.54及所有实验权重、预测和协议。下一阶段应优先核查时间与频率压缩是否损失微弱活动信息，并用同容量、同切分对照验证，避免继续叠加未经验证的门控。','','代码位于experiments/record_mix_20261109；大文件保存在D:/huawei2026-experiments下相应日期目录。']
 (ROOT/'docs/CONDITIONAL_ROUTES_20261109.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 reg=read(ROOT/'models/registry.json');reg.update(updated_date='2026-11-09',research_status='conditional_routes_and_energy_complete_no_export',record_mix_20261109=result,next_submission=None);save(ROOT/'models/registry.json',reg)
 p=ROOT/'README.md';s=p.read_text(encoding='utf-8-sig');prefix='最新研究（2026-11-09）：完成15次记录混合、条件分类与扰动强度恢复对照训练。未产生新的提交推荐，平台最佳仍0.54。见[本轮报告](docs/CONDITIONAL_ROUTES_20261109.md)。\n\n';p.write_text(prefix+s,encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
