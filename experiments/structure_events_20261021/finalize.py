"""Attach known validation limitations to every candidate identity record."""
from pathlib import Path
import shutil
from train import ROOT,WORK,read,save

def main():
    if (WORK/'platform_feedback_038.json').exists():raise RuntimeError('Subsequent platform0.38 feedback supersedes this pre-submission finalization; do not overwrite it.')
    selection=read(WORK/'selection.json');meta=read(WORK/'primary_candidate.json')
    meta.update(stress_gate_passed=selection['stress_gate_passed'],recommendation='research_only_not_replacement_for_platform037',known_limitations=dict(e1_stress_accuracy=0.0,e1_description='60 zero/one-person windows all predicted3',two_seed_four_person_recall=1/280,teacher_platform_score_unknown=True))
    for p in [WORK/'primary_candidate.json',ROOT/Path(meta['path']).with_suffix('.json'),ROOT/'submissions/structure_events_20261021/metadata.json']:save(p,meta)
    for p in [WORK/'audit.json',ROOT/'checks/structure_events_20261021.json',WORK/'summary.json']:
        v=read(p);v['primary']=meta;save(p,v)
    p=ROOT/'models/registry.json';r=read(p);r['structure_events_20261021'].update(stress_gate_passed=False,recommendation=meta['recommendation'],known_limitations=meta['known_limitations']);assert r['data_revision_20261021']['platform_score']==.35 and r['structured_v1']['score']==.37;save(p,r)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
    print('FINALIZED_RESEARCH_LIMITATIONS; platform_best=.37; prior_candidate=.35; new_candidate=unscored',flush=True)
if __name__=='__main__':main()
