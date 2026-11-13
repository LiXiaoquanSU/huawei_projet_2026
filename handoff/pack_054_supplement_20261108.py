"""Create an audited, additive handoff archive; do not modify model files."""
import json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'handoff/054_models_and_bias_supplement_20261108.zip'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 files={};generated={}
 def add(src,dst):
  p=Path(src);assert p.is_file(),p;assert dst not in files;files[dst]=p
 models=[('gprime','models/teammate054_20261025/gprime_full_group_fallback_web054.onnx','2b189c950412d0d770d82a0f118ebd95699e18b5b6706f56a8539c59a04b4534'),('selective','models/next053_20261025/01_g54_selective_lowcount.onnx','d4f5b05464a0754a105da21c1d991850abf001207b14e1295224075ea3f0ed41'),('stable','models/robust_bias054_20261025/01_selective054_bias_stable.onnx','9966149c7f122254a647e7a14921d3104b595b7fc7237c7b0e506c1bf2ad83cf')]
 for name,rel,digest in models:
  p=ROOT/rel;assert sha(p.read_bytes())==digest;add(p,'models/'+p.name)
 for sub in ['next053_20261025','robust_bias054_20261025']:
  for p in (ROOT/'models'/sub).glob('*.json'):add(p,f'evidence/{sub}/{p.name}')
 for name in ['NEXT053_20261025.md','ROBUST_BIAS054_20261025.md','TEAMMATE054_20261025.md']:
  add(ROOT/'docs'/name,'docs/'+name)
 for name in ['robust_search.py','repeat_search.py','evaluate_bias.py','build_robust.py','build_bias.py']:
  add(ROOT/'experiments/bias054_20261025'/name,'historical_source/'+name)
 dw=Path('D:/huawei2026-experiments/robust_bias054_20261025')
 for name in ['plan.json','folds.json','repeat_plan.json','repeat_results.json','selection_repeated.json','selection.json']:
  add(dw/name,'bias_search/'+name)
 a=json.loads(Path('D:/huawei2026-experiments/next053_20261025/export/all_predictions.json').read_text());b=json.loads((dw/'export/all_predictions.json').read_text())
 key=lambda x:(x['identifier'],x['start']);aa={key(x):x for x in a};bb={key(x):x for x in b};assert len(aa)==len(bb)==960 and aa.keys()==bb.keys();joined=[]
 for k in sorted(aa):
  x,z=aa[k],bb[k];assert x['truth']==z['truth'] and x['prediction']==z['base'];assert max(abs(v-w) for v,w in zip(x['logits'],z['base_logits']))<1e-4
  joined.append(dict(identifier=k[0],start=k[1],truth=x['truth'],gprime=x['base'],selective=x['prediction'],stable=z['prediction'],gprime_logits=x['base_logits'],selective_logits=x['logits'],stable_logits=z['logits']))
 differences=[x for x in joined if len({x['gprime'],x['selective'],x['stable']})>1]
 summary=dict(scope='960 seen labeled windows, training-source diagnostic, NOT hidden or held-out test',correct={k:sum(x[k]==x['truth'] for x in joined) for k in ['gprime','selective','stable']},disagreements=len(differences),pairs={})
 for p,q in [('gprime','selective'),('selective','stable'),('gprime','stable')]:summary['pairs'][p+'_to_'+q]=dict(changed=sum(x[p]!=x[q] for x in joined),corrected=sum(x[p]!=x['truth'] and x[q]==x['truth'] for x in joined),broken=sum(x[p]==x['truth'] and x[q]!=x['truth'] for x in joined))
 def js(name,obj):generated[name]=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
 js('predictions/all_960_comparison.json',joined);js('predictions/disagreements_only.json',differences);js('predictions/summary.json',summary)
 readme='''# 054模型与偏置补充包（2026-11-08）

先看本文件，再看 predictions/summary.json。三份 ONNX 的平台成绩均为用户报告的0.54；不是本次重新测分。旧报告正文中的“待测”属于历史状态，以本说明及各报告顶部最新反馈为准。

## 三份文件
1. models/gprime_full_group_fallback_web054.onnx：队友原 G′，约83.68MB。
2. models/01_g54_selective_lowcount.onnx：选择性适配版，约45.04MB。
3. models/01_selective054_bias_stable.onnx：此次索要的最终偏置版，约45.04MB。

第一版→第二版：原G′最终预测2/3/4时保留原输出，预测0/1时使用适配专家。专家更新相位/时间后缀，75%原参数+25%已训练更新参数；仍输出五类。同时合并共享计算并采用无新增舍入误差的存储压缩，文件变小不代表简单删除一半网络。
第二版→第三版：只在最终五维分数追加[-0.05,0,+0.15,-0.15,-0.15]，顺序0/1/2/3/4人。

## 五维偏置怎么估计
这里新增的是末尾残差，不是原G′内部Udirect偏置。固定1人残差为0，其他四维[-0.30,+0.30]、步长0.05，每底座28561组；比较半径0.1/0.2/0.3和两类目标。采用整记录三折及五组附加划分检查选参稳定性，加入房间及来源类别退步约束；最终选择选择性底座、半径0.2、准确率目标，再用全部开发记录拟合得到上述向量。详见docs/ROBUST_BIAS054_20261025.md及bias_search/。
这不是隐藏测试人数先验估计，不是全局最优证明。开发数据与教师已被多轮复用，记录级重复划分不能消除选择偏差。
原G′内部Udirect向量为[-0.33481252,-1.9802545,0.9744804,0.293164,1.0474226]；此外还有原末尾1人+0.35。原Udirect偏置的完整历史估计流程没有足够原始资料确认，本包不补造它。内部偏置参与路由，不能直接与末尾残差相加后移动到同一位置。

## 逐窗口比较
predictions/all_960_comparison.json：三模型对应的标签、预测和完整五维分数。
predictions/disagreements_only.json：仅保留三者存在预测差异的窗口。
predictions/summary.json：正确数及两两纠正/破坏数量。
这些是保存的实际原始输入ONNX运行记录，经identifier/start对齐，并核对两阶段公共模型的分数一致。全部960窗来自已见训练记录，只能诊断差异，不能作为泛化准确率或隐藏测试结果。

## 自己推理比较
安装 numpy 和 onnxruntime 后运行：
python compare_models.py input.npz --output result.json
NPZ应包含frames(float32,[600,1,3,1024,2])及intervals(float32,[600])。保持比赛规定的间隔单位和末尾填充约定。输出为三份模型的五维分数及argmax。

historical_source/是搜索/导出历史脚本，依赖原项目模块与D盘预测缓存，不是独立可运行的完整训练工程。ONNX本身可独立推理。evidence/含模型元数据、资源与结构核验；SHA256SUMS.json用于核对包内文件。

本包补充上一份037训练资料包，不含原始CSI/ATI、巨型搜索缓存，也不重复打包final.pt。没有混入2026-11-07未评分ordinal75候选。
'''
 generated['README.md']=readme.encode('utf-8')
 generated['compare_models.py']=('''import argparse,json
from pathlib import Path
import numpy as np
import onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,default=Path('comparison.json'));a=p.parse_args()
with np.load(a.input,allow_pickle=False) as d: feed={k:np.array(d[k],dtype=np.float32,copy=True) for k in ['frames','intervals']}
assert feed['frames'].shape==(600,1,3,1024,2) and feed['intervals'].shape==(600,)
assert all(np.isfinite(x).all() for x in feed.values())
options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
results={}
for path in sorted((Path(__file__).resolve().parent/'models').glob('*.onnx')):
 session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);z=session.run(None,feed)[0].reshape(5);assert np.isfinite(z).all();results[path.name]={'scores':z.tolist(),'prediction':int(z.argmax())};del session
a.output.write_text(json.dumps(results,indent=2)+'\\n',encoding='utf-8');print(json.dumps(results,indent=2))
''').encode('utf-8')
 manifest={k:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for k,p in files.items()};manifest.update({k:dict(bytes=len(v),sha256=sha(v)) for k,v in generated.items()});js('SHA256SUMS.json',manifest)
 assert not OUT.exists(),'Do not overwrite an existing handoff archive'
 with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for name,p in files.items():z.write(p,name)
  for name,data in generated.items():z.writestr(name,data)
 with zipfile.ZipFile(OUT) as z:
  assert z.testzip() is None
  for name,expected in manifest.items():data=z.read(name);assert len(data)==expected['bytes'] and sha(data)==expected['sha256'],name
 result=dict(path=str(OUT),bytes=OUT.stat().st_size,sha256=sha(OUT.read_bytes()),files=len(files)+len(generated),verified=True,comparison=summary);OUT.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
