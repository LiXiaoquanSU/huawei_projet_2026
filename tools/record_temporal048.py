"""Record tied .48 feedback while preserving both scored artifacts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL='models/temporal_rescue_20261024/01_temporal_consensus80.onnx'
HASH='e53e3a1c7ac8161f437679dd367a5ef27b4a9210c7c553d5e5f744b94d3fe526'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    meta=read(p.with_suffix('.json'));meta.update(platform_score=.48,status='scored_tied_best',
        source='user-reported platform result; same displayed0.48 as original guard',submission_id=None)
    save(p.with_suffix('.json'),meta)
    r=read(ROOT/'models/registry.json')
    r['research_candidates']=[meta if v.get('sha256')==HASH else v for v in r.get('research_candidates',[])]
    r['tied_current_best']=[v for v in r.get('tied_current_best',[]) if v.get('sha256')!=HASH]+[
        {k:meta[k] for k in ('id','path','sha256','platform_score','status','report')}]
    if r.get('next_submission',{} ) and r['next_submission'].get('sha256')==HASH:r['next_submission']=None
    r['research_status']='temporal_consensus80_scored048_tied'
    r['candidate_note']='Temporal80 also reported0.48. Keep original048 as control; no hidden per-class changes inferred.'
    save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[v for v in rows if v['sha256']==HASH];assert len(matches)==1
    matches[0].update(platform_score='0.48',source='user-reported platform result',status='scored_tied_best')
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print('Temporal80 recorded0.48; original048 control retained; no submission id supplied.')


if __name__=='__main__':main()
