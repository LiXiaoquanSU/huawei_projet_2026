"""Record .50-bias trial scored .48; keep the .35-bias .49 model best."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL='models/bias050_20261024/01_final_bias_one050.onnx'
HASH='0ee9fbf12e52fd03f2911269a321b5ea98a74843151d0b7a80be32b2fb4e772b'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    meta=read(p.with_suffix('.json'));meta.update(platform_score=.48,status='evaluated_below_best',
        source='user-reported platform result:0.48 after total1 bias0.50',submission_id=None)
    save(p.with_suffix('.json'),meta)
    r=read(ROOT/'models/registry.json');assert r['current_best']['platform_score']==.49
    r['research_candidates']=[meta if v.get('sha256')==HASH else v for v in r.get('research_candidates',[])]
    if r.get('next_submission') and r['next_submission'].get('sha256')==HASH:r['next_submission']=None
    r['research_status']='bias050_scored048_stop_bias_ladder'
    r['candidate_note']='Total1 bias0.35 scored0.49,0.50 scored0.48. Preserve0.49; stop monotone bias expansion, research representations.'
    r['latest_research']=dict(report='docs/AFTER_BIAS050_048.md',status='representation_research_no_training',best_scored=.49)
    save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[v for v in rows if v['sha256']==HASH];assert len(matches)==1
    matches[0].update(platform_score='0.48',source='user-reported platform result',status='evaluated_below_best')
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print('Recorded050-bias platform0.48;035-bias0.49 remains best. No model bytes changed.')


if __name__=='__main__':main()
