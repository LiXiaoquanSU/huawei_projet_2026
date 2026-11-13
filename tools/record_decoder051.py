"""Record user platform feedback without changing the scored ONNX."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='models/decoder_consensus_20261025/01_dual_decoder_consensus.onnx'
HASH='735a63b54294497923f3fe5a94dc5ed39bb03ee8b6f12c3ff2e93292ef84e63e'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH;m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));m.update(platform_score=.51,status='scored_below_best',source='user-reported platform0.51;2026-10-25;submissionIDunknown');save(p.with_suffix('.json'),m)
    p=ROOT/'models/registry.json';r=json.loads(p.read_text(encoding='utf-8'));r['research_candidates']=[m if x.get('sha256')==HASH else x for x in r.get('research_candidates',[])]
    if (r.get('next_submission') or {}).get('sha256')==HASH:r['next_submission']=None
    r['research_status']='decoder051_return_to054_bias';r['candidate_note']='Dual decoder fusion scored051 below054. Frozen054 output bias experiment next; both054 preserved.';save(p,r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[x for x in rows if x['sha256']==HASH];assert len(matches)==1;matches[0].update(platform_score='0.51',status=m['status'],source=m['source'])
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for filename in ('README.md','docs/SUMMARY.md','models/README.md','docs/DECODER_CONSENSUS_20261025.md'):
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1);notice='最新平台反馈：双解码器融合版0.51，低于最佳0.54；已取消待测推荐。回到两份已评分054比较最终输出偏置。以下旧状态为历史。'
        if notice not in text:p.write_text(heading+'\n\n'+notice+'\n'+rest,encoding='utf-8')
    print('Recorded0.51; both054models retained.')
if __name__=='__main__':main()
