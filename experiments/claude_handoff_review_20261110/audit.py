import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'handoff/claude_review_20261110/evidence'
def csvread(n):return list(csv.DictReader((P/n).open(encoding='utf-8-sig',newline='')))
def regression(d,y):
 X=np.c_[np.ones(len(d)),d];coef=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@coef;err=[]
 for i in range(len(y)):
  mask=np.arange(len(y))!=i;err.append(float(y[i]-X[i]@np.linalg.lstsq(X[mask],y[mask],rcond=None)[0]))
 return dict(n=len(y),intercept=float(coef[0]),slope=float(coef[1]),pearson=float(np.corrcoef(d,y)[0,1]),rmse=float(np.mean((pred-y)**2)**.5),loo_rmse=float(np.mean(np.square(err))**.5))
z=np.load(P/'submission_unlabeled_predictions.npz',allow_pickle=False);g=np.load(P/'gprime_unlabeled_logits.npy',allow_pickle=False).argmax(1);assert np.array_equal(g,z['gprime_pred']);d=(z['pred']!=g).mean(1);y=z['website_accuracy'];names=z['names'];scores=csvread('website_scores.csv');valid=[r for r in scores if r['website_accuracy'].strip()];vals=np.array([float(r['website_accuracy']) for r in valid]);allr=regression(d,y)
# Documented duplicate submission51907; retain original name, drop rt_control alias.
keep=names!='rt_control_s42';r=regression(d[keep],y[keep]);same=np.array_equal(z['pred'][names=='rt_control_s42'],z['pred'][names=='gprime_unitary_aug_verify_20261109__control_s42'])
local=csvread('labeled_rows_900.csv');labels=np.array([int(r['headcount']) for r in local]);q=np.load(P/'room_holdout_oof_900.npz',allow_pickle=False);oof={k:int((q[k].argmax(1)==labels).sum()) for k in q.files}
u=csvread('unlabeled_rows.csv');print('u_columns',u[0]);idkey=next(k for k in u[0] if k in ['record_id','identifier']);ids=np.array([r[idkey] for r in u]);record_cons=[np.bincount(g[ids==i],minlength=5).max()/sum(ids==i) for i in np.unique(ids)]
subgroups={}
for key,mask in [('near_Gprime_d_below_016',keep&(d<.16)),('far_d_above_04',keep&(d>.4))]:
 subgroups[key]=regression(d[mask],y[mask])
result=dict(csv_submissions=len(scores),scored=len(valid),max_score=float(vals.max()),count_max=int((vals==vals.max()).sum()),prediction_rows=len(names),unique_submissions_after_known_duplicate=int(keep.sum()),known_duplicate_predictions_identical=bool(same),all47_regression=allr,deduplicated46_regression=r,subgroups=subgroups,gprime_class_counts=np.bincount(g,minlength=5).tolist(),loro_counts=oof,loro_labels=np.bincount(labels,minlength=5).tolist(),record_majority_fraction_mean=float(np.mean(record_cons)),record_count=len(record_cons),limitations=['Source-reported platform scores, not independently checked on live platform.','Only hard predictions on U; no hidden test labels or paired correctness.','No weights or raw data in this handoff; no independent retraining.','Within-U records have correlated windows.'])
Path(__file__).with_name('audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
