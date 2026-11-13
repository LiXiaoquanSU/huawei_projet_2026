"""Record covariance auxiliary tied platform049; keep both exact scored files."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='models/correlation_consensus_20261024/01_correlation_consensus.onnx'
HASH='e86a2a17f324089d4c115d0dcec8ed5ab486247b21b615938e5596e5f6fed5e6'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    meta=read(p.with_suffix('.json'));meta.update(platform_score=.49,status='scored_tied_best',source='user-reported platform result:0.49 on2026-10-25',submission_id=None);save(p.with_suffix('.json'),meta)
    r=read(ROOT/'models/registry.json');assert r['current_best']['platform_score']==.49
    assert hashlib.sha256((ROOT/r['current_best']['path']).read_bytes()).hexdigest()==r['current_best']['sha256']
    r['research_candidates']=[meta if v.get('sha256')==HASH else v for v in r.get('research_candidates',[])]
    r['tied_best_049']=[v for v in r.get('tied_best_049',[]) if v.get('sha256') not in (HASH,r['current_best']['sha256'])]+[r['current_best'],meta]
    if (r.get('next_submission') or {}).get('sha256')==HASH:r['next_submission']=None
    r['updated_date']='2026-10-25';r['research_status']='correlation_scored049_tied_reassess_generalization'
    r['candidate_note']='Both original final-bias and covariance auxiliary scored0.49. Equal displayed score does not imply identical errors. Prefer original simpler reference; no pending candidate.'
    r['latest_research']=dict(report='docs/AFTER_CORRELATION049_20261025.md',status='next_round_analysis_no_new_training',best_scored=.49)
    save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[v for v in rows if v['sha256']==HASH];assert len(matches)==1
    matches[0].update(platform_score='0.49',source='user-reported platform result2026-10-25',status='scored_tied_best')
    with p.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    p=ROOT/'docs/CORRELATION_CONSENSUS_20261024.md';s=p.read_text(encoding='utf-8');notice='**2026-10-25平台反馈：本文件0.49，与原0.49并列；未突破。** 精确SHA已登记，提交编号未知。下文保留生成时状态。\n\n'
    if notice not in s:s=s.replace('## 状态',notice+'## 状态',1)
    p.write_text(s,encoding='utf-8')
    print('Recorded covariance049 tie. Both scored model bytes preserved.')
if __name__=='__main__':main()
