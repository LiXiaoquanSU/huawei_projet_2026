"""Record user platform053, keep teammate054 and the060 research target."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='models/g60_20261025/01_g54_encoder_quarter_seed42.onnx'
HASH='ac51e11fb24bb514f74fcf76d0f69c694e3c7e6909ec1c6cd6ab2bb7469bce27'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));m.update(platform_score=.53,status='scored_below_best',source='user-reported platform053; 2026-10-25; submissionIDunknown');save(p.with_suffix('.json'),m)
    r=json.loads((ROOT/'models/registry.json').read_text(encoding='utf-8'));assert r['current_best']['platform_score']==.54
    r['research_candidates']=[m if x.get('sha256')==HASH else x for x in r.get('research_candidates',[])]
    if (r.get('best_own') or {}).get('platform_score',0)<.53:r['best_own']=dict(m,status='best_own_below_team_best')
    if (r.get('next_submission') or {}).get('sha256')==HASH:r['next_submission']=None
    r['candidate_note']='Quarter encoder update scored053 below teammate054. Do not recommend as improvement. Next: actual preprocessing alignment and isolate presence versus conditional count.'
    r['research_status']='g60_candidate_scored053_reassess';save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[x for x in rows if x['sha256']==HASH];assert len(matches)==1;matches[0].update(platform_score='0.53',status='scored_below_best',source=m['source'])
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    p=ROOT/'docs/G60_EXPERIMENTS_20261025.md';s=p.read_text(encoding='utf-8');notice='**最新平台反馈：本候选0.53，低于队友0.54；本地缓存提升没有转化为平台提升。停止将其推荐为改进版。** 下文为生成时实验记录。\n\n'
    if notice not in s:heading,rest=s.split('\n',1);p.write_text(heading+'\n\n'+notice+rest,encoding='utf-8')
    for name in ('README.md','docs/SUMMARY.md','models/README.md'):
        p=ROOT/name;text=p.read_text(encoding='utf-8');notice='最新反馈：四分之一编码器更新版平台0.53，低于最佳0.54，已取消待提交推荐。目标仍为突破0.60。以下旧状态为历史记录。\n'
        if notice not in text:heading,rest=text.split('\n',1);p.write_text(heading+'\n\n'+notice+rest,encoding='utf-8')
    print('Recorded053. Best054 preserved; no pending candidate.')
if __name__=='__main__':main()
