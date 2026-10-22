"""Finish the queued external/ratio comparisons after official architecture training."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
import run
from run import WORK,FOLDS,read,save,summary
from net import ROOT

HERE=Path(__file__).resolve().parent
def command(file,*args):
    subprocess.run([sys.executable,'-X','utf8',str(HERE/file),*args],check=True)

def wait_for(paths,seconds=7200):
    start=time.monotonic()
    while not all(p.exists() for p in paths):
        if time.monotonic()-start>seconds:raise TimeoutError([str(p) for p in paths if not p.exists()])
        time.sleep(10)

if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    wait_for([WORK/'confirmation.json',WORK/'external/single_rx_features_v2.pt',WORK/'ratio_summary.pt'])
    choices=read(WORK/'selection_stage1.json')['selected']
    wait_for([WORK/'fits'/f"{r['variant']}_s42_full"/'metrics.json' for r in choices])
    print('OFFICIAL_STAGE_READY',flush=True)
    command('export_candidates.py')
    command('transfer.py')
    external=read(WORK/'external_transfer_summary.json');control=read(WORK/'source_pretrain_control_summary.json')
    baseline=summary('baseline',42)
    if external['selection_score']>max(baseline['selection_score'],control['selection_score']):
        d=run.load_data()
        for fold in FOLDS:
            run.fit(d,'external_transfer',2026,fold,WORK/'pretrain/external/final.pt')
            run.fit(d,'source_pretrain_control',2026,fold,WORK/'pretrain'/f'source_only_{fold}'/'final.pt')
        save(WORK/'external_confirmation.json',[summary(v,s) for v in ['external_transfer','source_pretrain_control'] for s in [42,2026]])
        # Preserve a research candidate even if seed confirmation is mixed; report that fact.
        command('transfer.py','--full')
        command('export_candidates.py','--variant','external_transfer')
    command('ratio.py')
    ratio=read(WORK/'ratio_summary.json')
    if ratio['selection_score']>baseline['selection_score']:
        import ratio as rnet
        d=run.load_data();d['s']=torch.cat([d['s'],torch.load(WORK/'ratio_summary.pt',weights_only=True)],1)
        run.fresh=lambda variant,seed:rnet.fresh_ratio(seed)
        for fold in FOLDS:run.fit(d,'ratio',2026,fold)
        save(WORK/'ratio_confirmation.json',[summary('ratio',s) for s in [42,2026]])
        run.fit(d,'ratio',42,'full')
        command('export_ratio.py')
    save(WORK/'execution_complete.json',dict(status='complete',official_screen_variants=8,extra_ratio_ablation=True,
          external_transfer_and_source_compute_control=True,old_platform_best=.37,new_platform_scores=None))
    print('ALL_EXPERIMENTS_COMPLETE',flush=True)
