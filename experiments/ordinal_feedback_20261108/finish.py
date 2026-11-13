import json,gc
from pathlib import Path
import numpy as np
import torch
from ratio_network import Model,base,ROOT,W as NET
HERE=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows]);splits=base.load(base.WORK/'protocol.json')['splits'];raws=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();diag=[]
 for variant in ['phase','joint']:
  x=torch.from_numpy(np.array(np.load(NET/f'{variant}.npy',mmap_mode='r'))).cuda()
  for fold,spec in splits.items():
   cp=torch.load(NET/f'{fold}_{variant}/final.pt',map_location='cpu',weights_only=False);m=Model().cuda().eval();m.load_state_dict(cp['model_state']);s=((raws-cp['summary_mean'].cuda())/cp['summary_std'].cuda()).clamp(-8,8);source=set(cp['fit_identifiers']);assert source.isdisjoint({rows[i]['identifier'] for i in spec['test']});preds={}
   with torch.no_grad():
    for name,indices in [('source_fit',spec['source']),('test',spec['test']),('static_only',spec['test']),('dynamic_only',spec['test'])]:
     out=[]
     for lo in range(0,len(indices),24):
      ii=indices[lo:lo+24];xx=x[ii]
      if name=='static_only':xx=xx.mean(-1,keepdim=True).expand_as(xx)
      elif name=='dynamic_only':xx=xx-xx.mean(-1,keepdim=True)
      out.append(m(xx,s[ii]).argmax(-1).cpu())
     pred=torch.cat(out);preds[name]=pred;diag.append(dict(fold=fold,variant=variant,mode=name,correct=int((pred==y[indices]).sum()),total=len(indices),prediction_changed_vs_test=None if name in ['source_fit','test'] else int((pred!=preds['test']).sum())))
   del m;gc.collect();torch.cuda.empty_cache()
  del x;torch.cuda.empty_cache()
 base.save(HERE/'ratio_dependency.json',diag)
 rng=base.load(base.BASE/'rng_pipeline_20261108/results.json');probes=base.load(base.BASE/'complex_ratio_probe_20261108/results.json');nets=base.load(NET/'results.json');totals={}
 for family,rs in [('rng',rng),('fixed_ratio',probes),('ratio_network',nets)]:
  for r in rs:totals.setdefault(family+'/'+r['variant'],[]).append(r['metrics']['correct'])
 sample_difference=[]
 for f in range(3):
  a=torch.load(base.BASE/f'rng_pipeline_20261108/record{f}_continue_rng/predictions.pt',map_location='cpu',weights_only=False);b=torch.load(base.BASE/f'rng_pipeline_20261108/record{f}_native_loader/predictions.pt',map_location='cpu',weights_only=False);sample_difference.append([sum(x!=y for x,y in zip(u,v)) for u,v in zip(a['sampled_indices'],b['sampled_indices'])])
 result=dict(date='2026-11-08',platform_feedback=dict(model='ordinal75_20261107',score=.47),platform_best=.54,next_submission=None,new_neural_fits=12,new_neural_epochs=102,new_linear_fits=9,local_totals={k:dict(per_fold=v,correct=sum(v),total=960) for k,v in totals.items()},sampler_changed_indices_per_epoch=sample_difference,ratio_dependency=diag,scope='Repeated development record folds; no hidden-label access; no new candidate exported')
 base.save(HERE/'results_summary.json',result)
 lines=['# 2026-11-08 后续实验结果','','ordinal75平台0.47已登记并撤销推荐。已测最佳保持0.54。新增12次神经网络训练（102轮）、9次线性探针；未找到有充分提交依据的新模型，目标0.60未完成。','','## 训练执行流程对照','','原037结构，seed42、5轮、相同数据划分和损失。近期重置随机流的对照422/960（145、109、168）。本轮不重置为427/960（134、137、156），恢复原生DataLoader/采样器为445/960（137、131、177）。后者第一折退步，另外两折提高；不能当作稳定、跨种子的改进。','','对照两种采样器的3折×5轮×640次抽样，仅1个索引不同；前两折全部相同。因此大部分差异不能归因于抽到了不同训练样本。DataLoader消耗CPU随机状态会影响MixStyle等操作的随机流，整体训练路径仍会变化。本轮不是完整复现原始037历史环境。','','## 原始CSI联合幅相对照','','受[FarSense](https://arxiv.org/abs/1907.03994)的CSI比值思路启发，构造有界坐标[2Re(hi·conj(hj)),2Im(hi·conj(hj)),|hi|²−|hj|²]/(|hi|²+|hj|²)，与仅相对相位对照。FarSense研究呼吸感知，本项目没有复现其呼吸算法，也没有把论文效果外推成人数识别成绩。新表示在有效载波、共享乘性扰动条件下抵消共同增益/相位；本地数值扰动检查通过，现实噪声和有效性阈值仍会影响效果。','','先做固定均值/标准差/差分统计探针，再做同容量共享天线对编码器，保留32频率×150时间点的轨迹。相位对照补零到相同通道数，两组均12轮、相同seed和采样预算。摘要归一化只在源记录拟合。','','|实验|各折正确/320|正确/960|准确率|','|---|---|---:|---:|']
 for k,v in totals.items():lines.append(f'|{k}|{v}|{sum(v)}|{sum(v)/960:.2%}|')
 lines+=['','## 表征依赖诊断','','static_only将每条通道的时间轨迹替换为其时间均值；dynamic_only减去该均值。两者保留原17维摘要，所以不是完整消除全部静态或动态信息。它们是输入干预，不是实际分布下的独立测试，不能据此认定因果机制。','','|结构|输入|正确/总数|相对正常留出预测改变数|','|---|---|---:|---:|']
 for v in ['phase','joint']:
  for mode in ['source_fit','test','static_only','dynamic_only']:
   rs=[r for r in diag if r['variant']==v and r['mode']==mode];c=sum(r['correct'] for r in rs);n=sum(r['total'] for r in rs);ch='—' if mode in ['source_fit','test'] else str(sum(r['prediction_changed_vs_test'] for r in rs));lines.append(f'|{v}|{mode}|{c}/{n} ({c/n:.2%})|{ch}|')
 lines+=['','source_fit跨3个模型累计1920次评价，同一来源记录会重复出现，不能当1920个独立样本；也不能当测试准确率。','','## 决定','','不继续微调ordinal75比例，不导出固定路径反事实版本。联合幅相的数值抗干扰性质没有转化为本次人数泛化收益，暂不扩大该表示的参数量或将它接入054。原生训练流程保留作以后复现实验的明确控制，但445/960不足以证明能超过平台054。','以后提交应要求更强的机制证据和跨设置稳定性，不能再把旧代理净增几个窗口作为主要推荐理由。本轮各表都是已复用开发数据，不能宣布任何方案已达到060。','','代码：`experiments/ordinal_feedback_20261108/`。大型特征、检查点和预测存D盘；原054与交接压缩包均未修改。']
 (ROOT/'docs/FOLLOWUP_ORDINAL047_20261108.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 reg=base.load(ROOT/'models/registry.json');reg['research_status']='ordinal047_followup_rng_and_joint_ratio_complete_no_export';reg['research_round_20261108']=result;reg['next_submission']=None;reg['updated_date']='2026-11-08';base.save(ROOT/'models/registry.json',reg)
 p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');prefix='本轮研究（2026-11-08）：ordinal75平台0.47，撤销推荐；新增12次神经训练及9次线性对照。原生训练流程445/960；联合幅相轨迹381/960，均无充分依据替代已测054。未导出新候选，见[后续实验](docs/FOLLOWUP_ORDINAL047_20261108.md)。\n\n';p.write_text(prefix+text,encoding='utf-8');print(json.dumps(dict(totals=result['local_totals'],dependency=diag),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
