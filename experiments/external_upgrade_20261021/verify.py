"""Independent data separation, cache/raw parity and invariance checks."""
import csv,json
from pathlib import Path
import blosc2
import numpy as np
import torch
from net import ROOT,fresh,bank_input
from prepare import WORK,save
from run import CACHE,read,sha
def csvrows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    torch.set_num_threads(4);s=read(WORK/'selection.json')
    cp=WORK/'fits'/f"{s['variant']}_s{s['seed']}_full/final.pt";state=torch.load(cp,map_location='cpu',weights_only=True);model=fresh(s['seed']).eval();model.load_state_dict(state['model_state']);del state
    d=torch.load(CACHE,map_location='cpu',weights_only=True);r=d['rows'][0];raw=Path('D:/huawei2026-data/labeled')
    c=blosc2.load_array(str(raw/f"CSI_{r['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{r['identifier']}.b2"))
    start,stop=int(r['start']),int(r['stop']);x=torch.from_numpy(c[start:stop].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[start:stop].copy());dt[-1]=0
    with torch.no_grad():
        t,summary=model.preprocess(x[None].double(),dt[None].double());t,su=t.float(),summary.float()
        te=float((t-d['t'][:1]).abs().max());se=float((su-d['s'][:1]).abs().max())
        assert te<.001 and se<.001,(te,se)
        z=model.forward_features(t,su);cached=model.forward_features(d['t'][:1],d['s'][:1]);ze=float((z-cached).abs().max());assert ze<.005,ze
        a=bank_input(t);h=model.bank(a);order=torch.arange(a.shape[1]-1,-1,-1);h2=model.bank(a[:,order]);pe=float((h-h2).abs().max());assert pe<1e-4,pe
        zero=model.bank(torch.zeros_like(a));assert torch.isfinite(zero).all()
    folds=[]
    for folder in sorted((WORK/'fits').iterdir()):
        if not (folder/'metrics.json').exists():continue
        m=read(folder/'metrics.json');tr=csvrows(folder/'train_manifest.csv');ids={r['identifier'] for r in tr}
        assert len(ids)==m['distinct_records_seen']==m['config']['training_records']
        if m['config']['fold']!='full':
            va=csvrows(folder/'validation_manifest.csv');assert ids.isdisjoint({r['identifier'] for r in va});assert all(r['room']!='unknown' for r in tr)
        else:assert len(ids)==48 and sum(r['room']=='unknown' for r in tr)==60
        if m['config']['variant']=='source_control':
            p=Path(m['config']['pretrain']).parent;info=read(p/'metrics.json');assert set(info['source_records'])==ids
        folds.append(dict(fit=folder.name,training_records=len(ids),distinct_records_seen=m['distinct_records_seen']))
    ext=torch.load(WORK/'external_sequences.pt',map_location='cpu',weights_only=True)
    assert len(ext['radio']['rows'])==797 and len(ext['sharp']['rows'])==485
    assert all(r['split']=='external_train' for r in ext['brescia']['rows']) and len(ext['brescia']['rows'])==518
    original=read(ROOT/'checks/restarts_cleanup_20261020.json')['model_hashes']
    for name,expected in original.items():assert sha(ROOT/name)==expected
    primary=read(WORK/'primary_candidate.json');alias=ROOT/'submissions/external_upgrade_20261021/model.onnx';assert sha(alias)==primary['sha256']
    result=dict(status='pass',cache_raw_feature_max_error=te,cache_raw_summary_max_error=se,cache_raw_logits_max_error=ze,bank_channel_permutation_max_error=pe,bank_zero_finite=True,room_and_record_separation=True,source_pretraining_uses_only_source_fold=True,all_full_fits_cover_48_records=True,brescia_date_holdout_not_trained=True,old_model_hashes_verified=len(original),fits=folds,primary=primary)
    save(WORK/'verification.json',result);save(ROOT/'checks/external_upgrade_20261021.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
