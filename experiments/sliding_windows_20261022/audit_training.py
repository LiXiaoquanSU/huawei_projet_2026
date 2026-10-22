"""Replay the dedicated sampler to audit actual windows and raw-frame coverage."""
import csv
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import WeightedRandomSampler
from common import ROOT,WORK,read,save,write_csv

def replay(rows,epochs,samples):
    per_record=defaultdict(int);group_records=defaultdict(set)
    domain=lambda r:r['identifier'] if r['room']=='unknown' else r['room']
    for r in rows:per_record[r['identifier']]+=1;group_records[(int(r['headcount']),domain(r))].add(r['identifier'])
    class_groups={c:len({g for k,g in group_records if k==c}) for c in range(5)}
    w=[1/(per_record[r['identifier']]*len(group_records[(int(r['headcount']),domain(r))])*class_groups[int(r['headcount'])]) for r in rows]
    sampler=WeightedRandomSampler(w,samples,generator=torch.Generator().manual_seed(42));seen=set()
    for _ in range(epochs):seen.update(iter(sampler))
    selected=[rows[i] for i in sorted(seen)];by_record=defaultdict(list)
    for r in selected:by_record[r['identifier']].append((int(r['start']),int(r['stop'])))
    coverage=[]
    for ident,intervals in by_record.items():
        covered=0;left=right=None
        for a,b in sorted(intervals):
            if left is None:left,right=a,b
            elif a<=right:right=max(right,b)
            else:covered+=right-left;left,right=a,b
        covered+=right-left;assert 0<covered<=12000
        coverage.append(dict(identifier=ident,actual_windows=len(intervals),covered_frames=covered,total_frames=12000,frame_coverage=covered/12000))
    return selected,coverage

def main():
    fits=list((WORK/'fits').glob('*/metrics.json'));assert len(fits)==33;results=[]
    for p in fits:
        m=read(p);cfg=m['config']
        with (p.parent/'train_manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        selected,coverage=replay(rows,cfg['epochs'],cfg['samples_per_epoch'])
        assert len(selected)==m['actual_windows'] and len(coverage)==m['actual_records']
        write_csv(p.parent/'sampled_windows.csv',selected);write_csv(p.parent/'sampled_record_coverage.csv',coverage)
        record=dict(fit=p.parent.name,actual_windows=len(selected),available_windows=len(rows),records=len(coverage),mean_raw_frame_coverage=sum(r['frame_coverage'] for r in coverage)/len(coverage),optimizer_updates=cfg['epochs']*cfg['optimizer_steps_per_epoch'],sampler_replay_matches_training=True)
        if cfg['fold']=='full':
            old='fresh_backbones_20261022' if cfg['architecture']=='mixer' else 'global_local_20261021';folder=Path('D:/huawei2026-experiments')/old/'fits'/f"{cfg['architecture']}_s{cfg['seed']}_full"
            with (folder/'train_manifest.csv').open(encoding='utf-8-sig') as f:oldrows=list(csv.DictReader(f))
            oldselected,oldcoverage=replay(oldrows,cfg['epochs'],len(oldrows));assert len(oldselected)==read(folder/'metrics.json')['actual_windows']
            record['old_full_raw_frame_coverage']=sum(r['frame_coverage'] for r in oldcoverage)/len(oldcoverage)
        save(p.parent/'sampling_audit.json',record);results.append(record)
    save(WORK/'sampling_audit.json',results);save(ROOT/'checks/sliding_sampling_20261022.json',dict(status='pass',fits=33,results=results));print([r for r in results if r['fit'].endswith('_full')])
if __name__=='__main__':main()
