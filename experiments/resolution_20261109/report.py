import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];D=Path('D:/huawei2026-experiments');W=D/'resolution_20261109'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 rs=read(W/'results.json');assert len(rs)==6
 prior=read(D/'dynamic_evidence_20261108/results.json');rs+=[r for r in prior if r['mode']=='raw'];rows=read(D/'next053_20261025/rows.json');agg={};pred={}
 for mode in ['raw','time300','freq64']:
  rr=[r for r in rs if r['mode']==mode];cm=np.array([r['metrics']['confusion'] for r in rr]).sum(0);assert cm.sum()==960
  agg[mode]=dict(correct=int(cm.trace()),total=960,accuracy=float(cm.trace()/960),fold_correct=[r['metrics']['correct'] for r in rr],confusion=cm.tolist(),recall=(cm.diagonal()/cm.sum(1)).tolist(),source_correct=sum(r['source_fit']['correct'] for r in rr),source_total=1920,parameters=rr[0]['parameters'])
  folder=D/'dynamic_evidence_20261108' if mode=='raw' else W
  pred[mode]=[torch.load(folder/f'record{f}_{mode}'/'predictions.pt',map_location='cpu',weights_only=False) for f in range(3)]
 contrasts={};errors=[]
 for mode in ['time300','freq64']:
  correct=broken=changed=0;deltas=[]
  for a,b in zip(pred[mode],pred['raw']):
   assert a['indices']==b['indices'] and torch.equal(a['labels'],b['labels']);y=a['labels'];aa=a['logits'].argmax(-1);bb=b['logits'].argmax(-1);delta=((aa==y).float()-(bb==y).float()).numpy();correct+=int(((aa==y)&(bb!=y)).sum());broken+=int(((aa!=y)&(bb==y)).sum());changed+=int((aa!=bb).sum());ids=np.array([rows[i]['identifier'] for i in a['indices']]);deltas.append(np.array([delta[ids==k].mean() for k in sorted(set(ids))]))
   for j,i in enumerate(a['indices']):
    errors.append(dict(mode=mode,index=i,identifier=rows[i]['identifier'],room=rows[i]['room'],true=int(y[j]),baseline=int(bb[j]),prediction=int(aa[j])))
  rng=np.random.default_rng(20261109);samples=[np.concatenate([rng.choice(d,len(d),replace=True) for d in deltas]).mean() for _ in range(10000)];contrasts[mode]=dict(corrected=correct,broken=broken,changed=changed,record_bootstrap95=np.quantile(samples,[.025,.975]).tolist())
 result=dict(date='2026-11-09',fits=6,epochs=72,aggregate=agg,contrasts=contrasts,cache_checks=read(W/'features_complete.json')|{'rows':'See D cache manifest'},platform_score=None,platform_best=.54,next_submission=None,status='complete_no_export')
 save(Path(__file__).with_name('results_summary.json'),result);save(Path(__file__).with_name('window_errors.json'),errors)
 lines=['# 输入分辨率对照：2026-11-09','','本轮完成6次训练、72轮。保留同一组完整记录三折，每折640个训练窗口、320个验证窗口。seed42、12轮、AdamW lr0.0003、批24、同人数×房间加权采样。外层记录不参与训练或选轮，但这些折已反复用于开发，结果不是独立盲测。','','## 实验设计','','原始输入为有界天线对联合幅相坐标，不做动态归一化，17维摘要全部置零。基线为频率32块×时间150点，本轮分别提高到32×300及64×150。卷积和分类器参数量不变；频率64组在卷积后将统计特征平均到4个频率位置，保证投影层相同。增加分辨率同时改变有效感受野与计算量，因此不能把结果解释成只增加信息量的纯因果效应。','','基线复用此前raw组。新网络在32×150上与旧网络同权重输出逐值一致；新时间缓存还原到150点，与旧缓存最大误差小于1e-6。缓存直接来自48条原始CSI，未读取验证标签进行变换。','','|方案|各折正确/320|总正确/960|准确率|训练拟合/1920|','|---|---|---:|---:|---:|']
 for m,v in agg.items():lines.append(f"|{m}|{v['fold_correct']}|{v['correct']}|{v['accuracy']:.2%}|{v['source_correct']}|")
 lines+=['','## 各人数召回','','|方案|0人|1人|2人|3人|4人|','|---|---:|---:|---:|---:|---:|']
 for m,v in agg.items():lines.append('|'+m+'|'+'|'.join(f'{x:.2%}' for x in v['recall'])+'|')
 lines+=['','## 相对基线的错误变化','']
 for m,v in contrasts.items():lines.append(f"- {m}：纠正{v['corrected']}，破坏{v['broken']}；整记录、折内重采样描述性95%差值区间：{v['record_bootstrap95']}。")
 lines+=['','区间不校正方案选择、多轮复用开发集的偏差。高训练拟合不能证明跨记录泛化。此处比较的是新研究架构的分辨率，不是对已有平台0.54模型的直接消融，不能据此判断旧模型的分辨率收益。','','完整逐窗口预测见同目录window_errors.json；权重、协议、缓存与预测保存于D:/huawei2026-experiments/resolution_20261109。平台0.54文件保持不变。']
 (ROOT/'docs/RESOLUTION_20261109.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
