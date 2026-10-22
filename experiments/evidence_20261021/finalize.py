import shutil
from pathlib import Path
from train import ROOT,read
def main():
    audit=read(ROOT/'checks/after_039_research.json');assert audit['complete_fits']==28
    note='本轮0.39后研究已完成28次新拟合：[完整结果](AFTER_039_RESULTS.md)。新增幅相交互基本持平、时间偏置及类别查询未提升；固定序数读出本地更好，推荐待测文件为 `submissions/ordinal_readout_20261021/model.onnx`（70.18MB，SHA d1489377…081c36a），平台分数未知。两份原0.39保持并列最佳。\n\n'
    for name in ('SUMMARY.md','EXPERIMENT_INDEX.md'):
        p=ROOT/'docs'/name;s=p.read_text(encoding='utf-8')
        if note not in s:s=s.replace('\n','\n\n'+note,1)
        p.write_text(s,encoding='utf-8')
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8');note2='本轮已完成28次新拟合，成功与失败结构、训练审计见[汇总报告](docs/AFTER_039_RESULTS.md)。\n\n'
    if note2 not in s:s=s.replace('\n','\n\n'+note2,1)
    p.write_text(s,encoding='utf-8')
    for tag in ('evidence_20261021','query_20261021','ordinal_readout_20261021'):
        work=Path('D:/huawei2026-experiments')/tag;snap=work/'final_source_snapshot';snap.mkdir(exist_ok=True)
        for p in Path(__file__).parent.iterdir():
            if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
        for name in ('src','experiments/global_local_20261021','experiments/structure_events_20261021'):
            target=snap/'dependencies'/name;target.mkdir(parents=True,exist_ok=True)
            for p in (ROOT/name).glob('*.py'):shutil.copy2(p,target/p.name)
        shutil.copy2(ROOT/'docs/AFTER_039_RESULTS.md',work/'CAMPAIGN_SUMMARY.md')
        shutil.copy2(ROOT/'checks/after_039_research.json',work/'campaign_audit.json')
    print('FINALIZED: 28 fits, two unscored exports, original platform best .39 preserved')
if __name__=='__main__':main()
