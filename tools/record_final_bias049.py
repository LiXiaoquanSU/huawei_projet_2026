"""Promote user's scored .49 exact artifact; preserve all previous models."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL='models/final_bias_20261024/01_final_bias_one035.onnx'
HASH='d083e3bb36f522e29b45acb88e41e5577013f9140b2f59dd2f9741c476f839df'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    meta=read(p.with_suffix('.json'));meta.update(id='best_049_final_bias',name='最终1人偏置加0.35',
        platform_score=.49,status='current_best',source='user-reported platform result:0.49',
        owner='self_final_bias_on_teammate_backbone',submission_id=None)
    save(p.with_suffix('.json'),meta);save(ROOT/'checks/best049_final_bias.json',meta)
    r=read(ROOT/'models/registry.json')
    old=r.get('current_best',{})
    if old.get('sha256')!=HASH:
        r['previous_tied_best_048']=[old,*r.get('tied_current_best',[])]
    best={k:meta[k] for k in ('id','name','path','sha256','bytes','platform_score','status','source','owner','submission_id','report')}
    for v in r['retained_models']:
        if v.get('platform_score')==.48:v['status']='retained_former_best'
    r['retained_models']=[v for v in r['retained_models'] if v.get('sha256')!=HASH]+[best]
    for v in r.get('research_candidates',[]):
        if v.get('platform_score')==.48:v['status']='retained_former_best'
    r['research_candidates']=[v for v in r.get('research_candidates',[]) if v.get('sha256')!=HASH]
    r['current_best']=best;r['best_own']=dict(best,ownership_note='Final residual bias only; underlying scored backbone unchanged.')
    r['tied_current_best']=[]
    if r.get('next_submission') and r['next_submission'].get('sha256')==HASH:r['next_submission']=None
    r['research_status']='final_bias_scored049'
    r['candidate_note']='User reports0.49 for final1+0.35. Target0.54 remains unmet. Next round is planning only.'
    r['latest_research']=dict(report='docs/NEXT_ROUTE_AFTER049.md',status='planning_only',best_scored=.49)
    save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[v for v in rows if v['sha256']==HASH];assert len(matches)==1
    matches[0].update(platform_score='0.49',source='user-reported platform result',status='current_best')
    for v in rows:
        if v['platform_score']=='0.48':v['status']='retained_former_best'
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for relative in ('models/original_one_guard_20261024/01_original_one_guard.json','models/temporal_rescue_20261024/01_temporal_consensus80.json'):
        p=ROOT/relative;v=read(p);v['status']='retained_former_best';save(p,v)
    print('Recorded0.49 as best; exact ONNX untouched; previous0.48/0.45 retained; target0.54.')


if __name__=='__main__':main()
