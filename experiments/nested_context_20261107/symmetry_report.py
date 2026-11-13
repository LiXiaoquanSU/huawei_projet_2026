import json
from pathlib import Path
import numpy as np
import torch
W=Path('D:/huawei2026-experiments/context_symmetry_20261107')
ROOT=Path(__file__).resolve().parents[2]
def main():
 rows=json.loads(Path('D:/huawei2026-experiments/next053_20261025/rows.json').read_text()); records=sorted({r['identifier'] for r in rows})
 correct={};cms={};perfold={}
 for v in ['identity','time_reverse','rx_time']:
  for fold in range(3):
   p=W/f'record{fold}_{v}/predictions.pt'
   if not p.exists(): print('INCOMPLETE',p);return
   d=torch.load(p,map_location='cpu',weights_only=False);z=d['raw'];y=d['labels'];ix=d['indices']
   for mode,prob in [('identity',z[:,0,0]),('time2',z[:,0].exp().mean(1)),('all12',z.exp().mean((1,2)))]:
    key=v+'/'+mode;pred=prob.argmax(1);ok=(pred==y).numpy();correct.setdefault(key,np.zeros(960,dtype=bool))[ix]=ok
    cms[key]=cms.get(key,np.zeros((5,5),dtype=int))+torch.bincount(y*5+pred,minlength=25).reshape(5,5).numpy();perfold.setdefault(key,[]).append(int(ok.sum()))
 rng=np.random.default_rng(20261107);draw=rng.integers(len(records),size=(10000,len(records)));baseline=correct['identity/identity'];result={}
 for key,ok in correct.items():
  delta=ok.astype(float)-baseline;group=np.array([delta[[i for i,r in enumerate(rows) if r['identifier']==rec]].mean() for rec in records]);ci=np.quantile(group[draw].mean(1),[.025,.975]).tolist()
  cm=cms[key];result[key]=dict(correct=int(ok.sum()),accuracy=float(ok.mean()),per_fold=perfold[key],confusion=cm.tolist(),class_recall=(np.diag(cm)/cm.sum(1)).tolist(),corrected=int((ok&~baseline).sum()),broken=int((~ok&baseline).sum()),record_bootstrap_delta95=ci)
 (Path(__file__).parent/'symmetry_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 lines=['# 2026-11-07：上下文模型的变换增强对照','','这不是平台成绩；三个记录折被多轮研究复用，区间也不纠正历史模型选择偏差。固定 seed42、5 轮、每轮源窗口数的采样预算，相同结构和损失。identity 权重复用上一项严格记录划分实验；其余 6 次为新训练。没有用外层标签调偏置、选轮或改变融合比例。','','|训练/推理|正确/960|准确率|各折/320|修正/破坏|相对原输入对照差值95%区间（记录重采样）|','|---|---:|---:|---|---|---|']
 for key,r in result.items():
  lo,hi=r['record_bootstrap_delta95'];lines.append(f"|{key}|{r['correct']}|{r['accuracy']:.2%}|{r['per_fold']}|{r['corrected']}/{r['broken']}|{lo:+.2%} 至 {hi:+.2%}|")
 lines+=['','修正/破坏均相对 identity 训练、identity 推理。接收通道标签变化只重新表达同一观测，不能增加独立房间或真实人数信息；随机变换可能减弱对通道编号的依赖，也可能抹掉训练分布中有用的设备方向关联，需由对照证据判断。','']
 (ROOT/'docs/CONTEXT_SYMMETRY_20261107.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))
if __name__=='__main__':main()
