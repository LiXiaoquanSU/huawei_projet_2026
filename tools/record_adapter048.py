"""Record user platform .48 for the exact temporal adapter candidate."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='models/temporal_adapter_20261024/01_temporal_adapter_consistent.onnx'
HASH='ec15efcd1341ee088b1796ab2839d0c01d217225ceed971509f59513c46ef904'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH
    meta=read(p.with_suffix('.json'));meta.update(platform_score=.48,status='evaluated_below_best',source='user-reported platform result:0.48',submission_id=None);save(p.with_suffix('.json'),meta)
    registry=read(ROOT/'models/registry.json');assert registry['current_best']['platform_score']==.49
    registry['research_candidates']=[meta if r.get('sha256')==HASH else r for r in registry.get('research_candidates',[])]
    if (registry.get('next_submission') or {}).get('sha256')==HASH:registry['next_submission']=None
    registry['research_status']='temporal_adapter_scored048_stop_source_adapter_scaling'
    registry['candidate_note']='Temporal adapter source688->718 but platform049->048. No pending candidate; preserve scored049.'
    save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    match=[r for r in rows if r['sha256']==HASH];assert len(match)==1
    match[0].update(platform_score='0.48',source='user-reported platform result',status='evaluated_below_best')
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    p=ROOT/'docs/TEMPORAL_ADAPTER_20261024.md';s=p.read_text(encoding='utf-8')
    notice='**后续平台反馈：本文件0.48，低于原0.49，停止推荐。** 精确SHA已登记；下文保留当时实验及提交理由。\n\n'
    if notice not in s:s=s.replace('## 当前结论与提交文件',notice+'## 当前结论与提交文件',1)
    p.write_text(s,encoding='utf-8')
    for name in ('README.md','docs/SUMMARY.md','models/README.md'):
        p=ROOT/name;s=p.read_text(encoding='utf-8');heading=s.splitlines()[0]
        notice='最新平台反馈：时间适配器版0.48，低于原0.49；源数据688→718没有迁移。已取消待提交状态，保持0.49为已评分最佳，继续研究固定权重的输入鲁棒性。\n\n'
        if notice not in s:s=s.replace(heading+'\n\n',heading+'\n\n'+notice,1)
        p.write_text(s,encoding='utf-8')
    print('Recorded adapter048; best049 unchanged')
if __name__=='__main__':main()
