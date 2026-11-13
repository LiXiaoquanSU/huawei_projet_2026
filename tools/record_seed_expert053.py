"""Record user feedback for the exact last exploratory model, retaining best054."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REL='models/seed_experts_20261025/02_exploratory_s310_adjacent_up_rx2.onnx'
SHA='885029329cf1e65e67011b005090f985ddc7d7dcf64ddfb5af75431bc21ff243'
PACKAGE=ROOT/'2026-10-25_037配方与种子复现'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def csvsave(p,fields,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=ROOT/REL;assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
    m=read(p.with_suffix('.json'));m.update(platform_score=.53,status='scored_below_best',source='user-reported platform0.53;2026-10-25;submissionIDunknown')
    save(p.with_suffix('.json'),m)
    copy=PACKAGE/'新候选模型'/p.name;assert hashlib.sha256(copy.read_bytes()).hexdigest()==SHA;save(copy.with_suffix('.json'),m)
    reg=read(ROOT/'models/registry.json');assert reg['current_best']['platform_score']==.54
    reg['research_candidates']=[m if x.get('sha256')==SHA else x for x in reg.get('research_candidates',[])]
    reg['latest_scored']=m;reg['research_status']='seed_expert053_temporal_origin_next';reg['candidate_note']='Crossseedordinalup scored053, below054 despite local+1.82pp. Preserve054; next direction temporalorigin marginalization withinoriginal weights.'
    if (reg.get('next_submission') or {}).get('sha256')==SHA:reg['next_submission']=None
    save(ROOT/'models/registry.json',reg)
    f=ROOT/'models/platform_scores.csv';fields,rr=csvrows(f);entry={k:'' for k in fields}
    entry.update(model='跨种子相邻人数向上修正，seed310双视角',platform_score='0.53',path=REL,sha256=SHA,source=m['source'],status=m['status'],availability='retained',original_path=REL)
    rr=[r for r in rr if r['sha256']!=SHA]+[entry];csvsave(f,fields,rr)
    f=PACKAGE/'platform_scores.csv';fields,rr=csvrows(f);r={k:'' for k in fields};r.update({'模型':entry['model'],'平台分':'0.53','说明':'本地留房间+1.82pp，平台低于原054；不能把本地提升当平台收益','sha256':SHA,'文件路径':REL,'来源':m['source'],'记录状态':'已测低于最佳'})
    r[fields[0]]='10-25';rr=[x for x in rr if x.get('sha256')!=SHA]+[r];csvsave(f,fields,rr)
    f=PACKAGE/'新增模型清单.csv';fields,rr=csvrows(f)
    for r in rr:
        if r['sha256']==SHA:r.update(平台分='0.53',状态='已测低于最佳054')
    csvsave(f,fields,rr)
    notice='\n\n最新平台反馈：本轮seed310双视角相邻向上修正得到 **0.53**，低于原0.54。以下“未测”及交付描述保留为历史记录。下一轮转向原权重内部的时间原点稳健性，不重复放大跨种子融合。\n'
    for rel in ['docs/SEED_EXPERTS_20261025.md','docs/TEAM_PROGRESS_FULL_20261025.md']:
        f=ROOT/rel;text=f.read_text(encoding='utf-8')
        if notice.strip() not in text:heading,rest=text.split('\n',1);f.write_text(heading+notice+rest,encoding='utf-8')
    for src,dest in [('docs/SEED_EXPERTS_20261025.md','下一轮实验_跨种子条件专家.md'),('docs/TEAM_PROGRESS_FULL_20261025.md','团队完整总结_含小全实验.md')]:
        (PACKAGE/dest).write_text((ROOT/src).read_text(encoding='utf-8'),encoding='utf-8')
    for f in [PACKAGE/'README.md',PACKAGE/'新候选模型/先读说明.md']:
        text=f.read_text(encoding='utf-8')
        if notice.strip() not in text:heading,rest=text.split('\n',1);f.write_text(heading+notice+rest,encoding='utf-8')
    print('Recorded exact model053; best054 unchanged; team documents updated.')

if __name__=='__main__':main()
