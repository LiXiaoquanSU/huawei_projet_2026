"""Surgical054 graph change and independent arithmetic audit on raw CSI."""
import json,hashlib,gc
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort,blosc2,torch
from onnx import helper,numpy_helper
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'models/teammate054_20261025/gprime_full_group_fallback_web054.onnx';W=Path('D:/huawei2026-experiments/ordinal_policy_20261107/export');SHA='2b189c950412d0d770d82a0f118ebd95699e18b5b6706f56a8539c59a04b4534'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def policy(p,b):
 z=np.log(np.maximum(p,1e-8));v0=z[0,0]+b;v45=np.log(np.exp(z[0]).mean(0))+b;fallback=np.log(np.exp(np.log(np.exp(z).mean(1))+b).mean(0));out=v0 if v0.argmax()==1 else v45 if v45.argmax() in (0,2) else fallback
 return out+np.array([0,.35,0,0,0],dtype=np.float32)
def main():
 W.mkdir(parents=True,exist_ok=True);assert sha(SOURCE)==SHA;m=onnx.load(SOURCE);orig=onnx.ModelProto();orig.CopyFrom(m);nodes=[];changed=[]
 direct_name='__ordinal75_direct_weight';ordinal_name='__ordinal75_order_weight'
 m.graph.initializer.extend([numpy_helper.from_array(np.array(.25,dtype=np.float32),direct_name),numpy_helper.from_array(np.array(.75,dtype=np.float32),ordinal_name)])
 for node in m.graph.node:
  if node.name.endswith('/node_mul_59'):
   assert node.op_type=='Mul';prefix=node.output[0].removesuffix('mul_59');assert list(node.input)==[prefix+'add_49',prefix+'val_5'];changed.append(node.name)
   nodes.extend([helper.make_node('Mul',[prefix+'add_48',direct_name],[prefix+'new_direct25'],name=prefix+'weight_direct25'),helper.make_node('Mul',[prefix+'cat_21',ordinal_name],[prefix+'new_ordinal75'],name=prefix+'weight_ordinal75'),helper.make_node('Add',[prefix+'new_direct25',prefix+'new_ordinal75'],list(node.output),name=prefix+'fuse_ordinal75')])
  else:nodes.append(node)
 assert len(changed)==12;del m.graph.node[:];m.graph.node.extend(nodes)
 assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(orig.graph.initializer,m.graph.initializer))
 kept={n.name:n.SerializeToString() for n in m.graph.node if n.name}
 assert all(not n.name or n.name in changed or kept[n.name]==n.SerializeToString() for n in orig.graph.node)
 helper.set_model_props(m,dict(base_sha256=SHA,base_platform_score='0.54 user-reported',platform_score='unknown',change='Per-view probability mixture .75 ordinal + .25 direct; original weights, Udirect bias, twelve views, protection formulas and final bias byte-identical',evidence='Four leave-room proxies 278 to284 of880; seen-record diagnostic676 to679 of960; NOT hidden platform accuracy',limitation='Small reused-development gain; two-person recall tradeoff; no guarantee of platform improvement'))
 onnx.checker.check_model(m);onnx.save(m,W/'model.onnx');save(W/'structure_proof.json',dict(base_sha256=SHA,candidate_sha256=sha(W/'model.onnx'),replaced_nodes=changed,all_original_initializers_byte_identical=True,all_other_original_nodes_preserved=True,bytes=(W/'model.onnx').stat().st_size))
 # Expose unmodified direct and ordinal outputs for an independent policy calculation.
 prefixes=[f'__rxg_{p}___tta_v{d}_' for p in ['012','021','102','120','201','210'] for d in [0,1]]
 for prefix in prefixes:
  for suffix in ['add_48','cat_21']:orig.graph.output.append(helper.make_tensor_value_info(prefix+suffix,onnx.TensorProto.FLOAT,[1,5]))
 opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
 ref=ort.InferenceSession(orig.SerializeToString(),sess_options=opt,providers=['CPUExecutionProvider']);candidate=ort.InferenceSession(str(W/'model.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
 bias=next(numpy_helper.to_array(t) for t in orig.graph.initializer if t.name=='__roomsense_bias_0_constant');del m,orig;gc.collect()
 rows=json.loads(Path('D:/huawei2026-experiments/next053_20261025/rows.json').read_text());pred=torch.load(W.parent/'actual/predictions.pt',map_location='cpu',weights_only=False)['predictions'];basepred=pred['baseline'].argmax(1).numpy();newpred=pred['ordinal75'].argmax(1).numpy();chosen={i for i,r in enumerate(rows) if r['start']==0}|set(np.where(basepred!=newpred)[0].tolist());records=[];references=[]
 def check(feed,index):
  z,*parts=ref.run(None,feed);p=np.array([v.reshape(5) for v in parts]).reshape(6,2,2,5);direct=p[:,:,0];ordinal=p[:,:,1];expected=policy(.25*direct+.75*ordinal,bias);original_expected=policy(.5*direct+.5*ordinal,bias);actual=candidate.run(['logits'],feed)[0]
  err=float(np.abs(actual-expected).max());baseerr=float(np.abs(z-original_expected).max());assert np.isfinite(actual).all() and err<1e-4 and baseerr<1e-4,(index,err,baseerr)
  row=dict(index=index,baseline=int(z.argmax()),candidate=int(actual.argmax()),max_error=err,baseline_formula_error=baseerr)
  if isinstance(index,int):row.update(truth=rows[index]['headcount'],identifier=rows[index]['identifier'],start=rows[index]['start'],cached_baseline=int(basepred[index]),cached_candidate=int(newpred[index]))
  records.append(row);return expected
 for identifier in sorted({rows[i]['identifier'] for i in chosen}):
  raw=Path('D:/huawei2026-data/labeled');c=blosc2.load_array(str(raw/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{identifier}.b2')).reshape(-1)
  for i in sorted(chosen):
   row=rows[i]
   if row['identifier']!=identifier:continue
   a,b=int(row['start']),int(row['stop']);feed=dict(frames=np.ascontiguousarray(c[a:b]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[a:b].astype(np.float32,copy=True));feed['intervals'][-1]=0;expected=check(feed,i)
   if len(references)<3 and (basepred[i]!=newpred[i] or not references):
    path=W/f'raw_reference_{i}.npz';np.savez_compressed(path,**feed);references.append(dict(input=str(path),logits=expected.tolist()))
  print('AUDIT',identifier,len(records),'/',len(chosen),flush=True);del c,ati;gc.collect()
 for path in [ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero']:
  if path=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
  else:
   with np.load(path) as d:feed={k:d[k].copy() for k in ['frames','intervals']}
  expected=check(feed,str(path));references.append(dict(input=str(path),logits=expected.tolist()))
 save(W/'references.json',references);save(W/'raw_audit.json',dict(status='pass',cases=records,max_error=max(x['max_error'] for x in records),scope='All cached-changed windows plus first window of each of48 records and smoke/gap/zero. Original ONNX intermediate probabilities independently recombined in numpy; exact original float preprocessing retained. These are numerical checks and seen-data diagnostics, not held-out validation.'));print('COMPLETE',flush=True)
if __name__=='__main__':main()
