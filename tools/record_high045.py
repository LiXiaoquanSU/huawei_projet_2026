"""Record user's .45 result without rewriting the submitted model."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL='models/high_count_20261024/01_conditional_high_count.onnx'
HASH='89f336e1682bb3226d1d99ef5b979bcb6b5d397c40d9fcf79ff4b82636193617'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    meta=read(p.with_suffix('.json'));meta.update(platform_score=.45,status='evaluated_below_best',
        source='user-reported platform result:0.45',submission_id=None)
    save(p.with_suffix('.json'),meta)
    r=read(ROOT/'models/registry.json');assert r['current_best']['platform_score']==.48
    r['research_candidates']=[meta if v.get('sha256')==HASH else v for v in r.get('research_candidates',[])]
    if r.get('next_submission') and r['next_submission'].get('sha256')==HASH:r['next_submission']=None
    r['research_status']='high_count_scored045_rejected'
    r['candidate_note']='High-count source fit improved654->688 but platform048->045; retain048 control. No hidden class-specific claims.'
    save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[v for v in rows if v['sha256']==HASH];assert len(matches)==1
    matches[0].update(platform_score='0.45',source='user-reported platform result',status='evaluated_below_best')
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print('Recorded high-count0.45; both0.48 artifacts retained unchanged.')


if __name__=='__main__':main()
