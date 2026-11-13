"""Record platform failure and decompose score versus routing changes."""
import sys,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as old
W=Path(__file__).resolve().parent
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def branch(raw,b,T):
 v0=raw[:,0,0]/T+b;v45=raw[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b
 return torch.where(v0.argmax(-1)==1,0,torch.where((v45.argmax(-1)==0)|(v45.argmax(-1)==2),1,2))
def scores(raw,b,T,r):
 v0=raw[:,0,0]/T+b;v45=raw[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b;fallback=(raw.exp().mean(2).clamp_min(1e-12).log()/T+b).exp().mean(1).clamp_min(1e-12).log()
 return torch.where((r==0)[:,None],v0,torch.where((r==1)[:,None],v45,fallback))+raw.new_tensor([0,.35,0,0,0])
def main():
 path=ROOT/'models/ordinal_head_20261107/model.json';meta=old.read(path);assert hashlib.sha256((ROOT/meta['path']).read_bytes()).hexdigest()==meta['sha256'];meta.update(platform_score=.47,status='scored_not_recommended',platform_feedback_date='2026-11-08',source='User-reported platform score0.47; submission ID not supplied');save(path,meta)
 registry=old.read(ROOT/'models/registry.json')
 def update(x):
  if isinstance(x,dict):
   if x.get('id')==meta['id']:x.update(platform_score=.47,status='scored_not_recommended',platform_feedback_date='2026-11-08')
   for v in x.values():update(v)
  elif isinstance(x,list):
   for v in x:update(v)
 update(registry)
 if isinstance(registry.get('next_submission'),dict) and registry['next_submission'].get('id')==meta['id']:registry['next_submission']=None
 registry['updated_date']='2026-11-08';registry['research_status']='ordinal75_platform047_recorded_followup_in_progress';save(ROOT/'models/registry.json',registry)
 feedback='最新平台反馈（2026-11-08）：该候选实测0.47，低于原0.54，已撤销推荐。下面的本地增益只属历史开发结果，不是平台提升。\n\n'
 for p in [ROOT/'README.md',ROOT/'docs/ORDINAL75_CANDIDATE_20261107.md']:
  text=p.read_text(encoding='utf-8-sig')
  if not text.startswith(feedback):p.write_text(feedback+text,encoding='utf-8')
 result=[]
 for room in ['a1','c1','g1','e1','actual']:
  d=torch.load(Path('D:/huawei2026-experiments/ordinal_policy_20261107')/room/'predictions.pt',map_location='cpu',weights_only=False);a=d['raw']['baseline'];n=d['raw']['ordinal75'];y=d['labels'];b,T=old.calibration(room);ra=branch(a,b,T);rn=branch(n,b,T);baseline=scores(a,b,T,ra);baseok=baseline.argmax(-1)==y
  for name,raw,r in [('baseline',a,ra),('score_only_old_route',n,ra),('route_only_old_score',a,rn),('actual_failed_candidate',n,rn)]:
   z=scores(raw,b,T,r);pred=z.argmax(-1);ok=pred==y;cm=torch.bincount(y*5+pred,minlength=25).reshape(5,5);row=dict(room=room,method=name,correct=int(ok.sum()),total=len(y),corrected=int((~baseok&ok).sum()),broken=int((baseok&~ok).sum()),route_changed=int((ra!=rn).sum()),confusion=cm.tolist());result.append(row);print(row,flush=True)
 save(W/'routing_diagnostic.json',dict(platform_feedback=meta,scope='Counterfactual diagnosis, not submitted or newly tuned candidates; actual denotes seen training records. Hidden test errors unavailable.',results=result));print('COMPLETE',flush=True)
if __name__=='__main__':main()
