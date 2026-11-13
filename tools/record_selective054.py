"""Record selective lowcount054, tied with teammate054; retain exact artifacts."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='models/next053_20261025/01_g54_selective_lowcount.onnx'
HASH='d4f5b05464a0754a105da21c1d991850abf001207b14e1295224075ea3f0ed41'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==HASH;m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));m.update(platform_score=.54,status='scored_tied_best',source='user-reported platform054;2026-10-25;submissionIDunknown');save(p.with_suffix('.json'),m)
    reg=json.loads((ROOT/'models/registry.json').read_text(encoding='utf-8'));assert reg['current_best']['platform_score']==.54
    reg['research_candidates']=[m if x.get('sha256')==HASH else x for x in reg.get('research_candidates',[])];reg['best_own']=m
    reg['tied_current_best']=[reg['current_best'],m];reg['tied_best_054']=[reg['current_best'],m]
    if (reg.get('next_submission') or {}).get('sha256')==HASH:reg['next_submission']=None
    reg['research_status']='selective054_tied_new_training_round';reg['candidate_note']='Selective lowcount scored054, tied with originalGprime. Retain both; compact selective version is new research control. Target060 not achieved.';save(ROOT/'models/registry.json',reg)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    matches=[x for x in rows if x['sha256']==HASH];assert len(matches)==1;matches[0].update(platform_score='0.54',status='scored_tied_best',source=m['source'])
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for filename in ('docs/NEXT053_20261025.md','README.md','docs/SUMMARY.md','models/README.md'):
        p=ROOT/filename;text=p.read_text(encoding='utf-8');notice='最新平台反馈：选择性低人数更新版0.54，与原Gprime0.54并列；两份原文件保留，0.60尚未达到。以下旧待测状态为历史记录。\n'
        if notice not in text:heading,rest=text.split('\n',1);p.write_text(heading+'\n\n'+notice+rest,encoding='utf-8')
    print('Recorded selective054tie; compact scored file and originalGprime preserved.')
if __name__=='__main__':main()
