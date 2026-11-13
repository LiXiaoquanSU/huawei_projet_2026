"""Read-only evidence audit: data support, paired complementarity, calibration leverage."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as old
BASE=Path('D:/huawei2026-experiments');OUT=ROOT/'experiments/feasibility_20261107'
def key(r):return (r['identifier'],r['start'],r['stop'])
def main():
 torch.set_num_threads(4);rows=old.read(BASE/'next053_20261025/rows.json');records={r['identifier']:r for r in rows};coverage={room:[sum(r['room']==room and int(r['headcount'])==c for r in records.values()) for c in range(5)] for room in sorted({r['room'] for r in rows})};results=[]
 for room in ('a1','c1','g1','e1'):
  d=torch.load(BASE/f'g60_20261025/{room}.pt',weights_only=True);b,T=old.calibration(room);fi,cal,te=old.split(room,d['rows']);base=old.policy(d['raw'],b,T);ix=torch.tensor(te);y=d['labels'][ix];bp=base[ix].argmax(-1);experts={};checked=[]
  for name,folder,seed in [('small42','evidence_chain_20261028',42),('small2026','evidence_chain_20261028',2026),('large42','evidence_large_20261028',42)]:
   x=torch.load(BASE/f'{folder}/{room}_full_s{seed}/predictions.pt',weights_only=False,map_location='cpu');lookup={key(r):i for i,r in enumerate(x['rows'])};jx=[lookup[key(d['rows'][i])] for i in te]
   assert set(jx)==set(x['test_indices']);assert torch.equal(y,x['labels'][jx]);assert not {x['rows'][i]['identifier'] for i in x['fit_indices']} & {d['rows'][i]['identifier'] for i in te}
   experts[name]=x['logits'][jx].argmax(-1);checked.append(name)
  baseok=bp==y
  for name,ep in experts.items():
   eo=ep==y;results.append(dict(room=room,expert=name,total=len(y),base_correct=int(baseok.sum()),expert_correct=int(eo.sum()),recoverable=int((~baseok&eo).sum()),at_risk=int((baseok&~eo).sum()),both_wrong=int((~baseok&~eo).sum()),disagree=int((ep!=bp).sum()),oracle_union=int((baseok|eo).sum()),by_class=[dict(label=c,total=int((y==c).sum()),recoverable=int((~baseok&eo&(y==c)).sum()),at_risk=int((baseok&~eo&(y==c)).sum())) for c in range(5)]))
  oracle=baseok.clone()
  for ep in experts.values():oracle|=ep==y
  results.append(dict(room=room,expert='all_three_oracle',total=len(y),base_correct=int(baseok.sum()),oracle_union=int(oracle.sum()),recoverable=int((~baseok&oracle).sum())))
 aggregate={}
 for name in ('small42','small2026','large42','all_three_oracle'):
  rr=[r for r in results if r['expert']==name];keys=['total','base_correct','recoverable','oracle_union']+([] if name=='all_three_oracle' else ['expert_correct','at_risk','both_wrong','disagree']);aggregate[name]={k:sum(r[k] for r in rr) for k in keys}
 calibration=[]
 for variant in ('identity_train','permuted_train'):
  for views in (1,6):
   for fold in range(3):
    d=torch.load(BASE/f'aligned_views_20261102/record{fold}_{variant}_v{views}_predictions.pt',weights_only=False,map_location='cpu');manifest=old.read(BASE/'detail_features_20261027/probe_results.json')['folds'][f'record{fold}'];ids=set(manifest['calibration_records']);ci=[i for i,r in enumerate(rows) if r['identifier'] in ids];y=torch.tensor([int(r['headcount']) for r in rows]);a=d['logits'];z=d['last_logits'];per_record=[]
    for ident in sorted(ids):
     ii=[i for i in ci if rows[i]['identifier']==ident];delta=float((-z[ii,y[ii]]+a[ii,y[ii]]).mean());per_record.append(dict(identifier=ident,label=int(y[ii[0]]),last_minus_selected_nll=delta))
    deltas=np.array([v['last_minus_selected_nll'] for v in per_record]);te=d['test_indices'];calibration.append(dict(fold=fold,variant=variant,views=views,cal_records=len(ids),cal_records_per_class=[sum(v['label']==c for v in per_record) for c in range(5)],cal_delta=float(deltas.mean()),drop_one_delta=[float(np.delete(deltas,k).mean()) for k in range(len(deltas))],test_selected=int((a[te].argmax(-1)==y[te]).sum()),test_last=int((z[te].argmax(-1)==y[te]).sum()),records=per_record))
 for r in calibration:
  d=torch.load(BASE/f'aligned_views_20261102/record{r["fold"]}_{r["variant"]}_v{r["views"]}_predictions.pt',weights_only=False,map_location='cpu');ids={v['identifier'] for v in r['records']};ci=[i for i,v in enumerate(rows) if v['identifier'] in ids];te=d['test_indices'];a=d['logits'];z=d['last_logits'];r['cal_selected_correct']=int((a[ci].argmax(1)==y[ci]).sum());r['cal_last_correct']=int((z[ci].argmax(1)==y[ci]).sum());r['test_last_minus_selected_nll']=float((-z[te,y[te]]+a[te,y[te]]).mean())
 output=dict(coverage=coverage,complementarity=results,aggregate=aggregate,calibration=calibration,limitations='No new training. All complementarity uses room-excluded teachers matched by identifier/start/stop; these are proxies, not actual054. Oracle uses ground truth and cannot deploy. Reused development data; calibration drop-one omits an entire class and is leverage diagnosis only, not valid five-class selection.')
 (OUT/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('COVERAGE',coverage);print('COMPLEMENT',aggregate);print('CALIBRATION',[(r['fold'],r['variant'],r['views'],round(r['cal_delta'],3),sum(v<0 for v in r['drop_one_delta']),r['test_selected'],r['test_last']) for r in calibration])
if __name__=='__main__':main()
