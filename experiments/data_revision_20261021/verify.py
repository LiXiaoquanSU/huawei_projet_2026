"""Check source identities, room separation, shape, topology and raw-input parity."""
import csv,json,zlib
from pathlib import Path
import blosc2
import numpy as np
import torch
from train import ROOT,WORK,CACHE,StructuredModel,read,sha
from acquire import save
def rows(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    torch.set_num_threads(4);sel=read(WORK/'selection.json');primary=read(WORK/'primary_candidate.json');cp=WORK/'fits'/f"{sel['variant']}_s{sel['seed']}_full/final.pt"
    state=torch.load(cp,map_location='cpu',weights_only=True);model=StructuredModel('spectral','physical').eval();assert sum(p.numel() for p in model.parameters())==19706024;assert set(model.state_dict())==set(state['model_state']);model.load_state_dict(state['model_state']);del state
    d=torch.load(CACHE,map_location='cpu',weights_only=True);r=d['rows'][0];raw=Path('D:/huawei2026-data/labeled');c=blosc2.load_array(str(raw/f"CSI_{r['identifier']}.b2"));ati=blosc2.load_array(str(raw/f"ATI_{r['identifier']}.b2"));a,b=int(r['start']),int(r['stop'])
    x=torch.from_numpy(c[a:b].view(np.float32).reshape(600,1,3,1024,2).copy());dt=torch.from_numpy(ati[a:b].copy());dt[-1]=0
    with torch.no_grad():
        t,s=model.preprocess(x[None].double(),dt[None].double());t,s=t.float(),s.float();te=float((t-d['t'][:1]).abs().max());se=float((s-d['s'][:1]).abs().max());ze=float((model.forward_features(t,s)-model.forward_features(d['t'][:1],d['s'][:1])).abs().max())
    assert te<.001 and se<.001 and ze<.005,(te,se,ze)
    data=torch.load(WORK/'external_dynamic.pt',map_location='cpu',weights_only=True)
    assert data['ehu_train']['x'].shape[2]==128 and data['ehu_test']['x'].shape[2]==128 and data['radio']['x'].shape[2]==30
    assert {r['room'] for r in data['ehu_train']['rows']}.isdisjoint({r['room'] for r in data['ehu_test']['rows']})
    assert {r['record'] for r in data['ehu_train']['rows']}.isdisjoint({r['record'] for r in data['ehu_test']['rows']})
    assert {r['room'] for r in data['ehu_test']['rows']}=={'Office 1'}
    for key,v in data.items():assert torch.isfinite(v['x']).all() and set(v['y'].tolist())==set(range(5))
    verified=[]
    for folder in sorted((WORK/'fits').iterdir()):
        if not (folder/'metrics.json').exists():continue
        m=read(folder/'metrics.json');tr=rows(folder/'train_manifest.csv');ids={r['identifier'] for r in tr}
        if m['config']['fold']=='full':assert m['actual_records']==len(ids)==48
        else:
            va=rows(folder/'validation_manifest.csv');assert ids.isdisjoint({r['identifier'] for r in va});assert all(r['room']!='unknown' for r in tr)
        if m['config']['pretrain'] and m['config']['variant']=='source_transfer':
            pre=rows(Path(m['config']['pretrain']).parent/'source_train_manifest.csv');assert {r['identifier'] for r in pre}==ids
        verified.append(dict(fit=folder.name,records=len(ids),actual_records=m['actual_records']))
    for p in (WORK/'pretrain').glob('ehu*/ehu_train_manifest.csv'):
        rr=rows(p);assert 'Office 1' not in {r['room'] for r in rr}
    root=WORK/'ehunam';selection=read(root/'selection.json');manifest={r['member']:r for r in read(root/'download_manifest.json')};total=0
    for member in selection['members']:
        m=manifest[member];path=Path(m['path']);payload=path.read_bytes();assert len(payload)==m['bytes'] and sha(path)==m['sha256'] and zlib.crc32(payload)&0xffffffff==m['crc32'];total+=len(payload)
    old=read(ROOT/'checks/restarts_cleanup_20261020.json')['model_hashes']
    for name,expected in old.items():assert sha(ROOT/name)==expected
    expected034='106ae34d405e651fc91230e9df17a44a338e8b0a00c4f88d5df86af19a4e8349';assert sha(ROOT/'submissions/external_upgrade_20261021/model.onnx')==expected034
    registry=read(ROOT/'models/registry.json');assert registry['external_upgrade_20261021']['platform_score']==.34
    alias=ROOT/'submissions/data_revision_20261021/model.onnx';assert sha(alias)==primary['sha256']
    result=dict(status='pass',original_topology_and_parameter_count=True,parameters=19706024,raw_cache_feature_max_error=te,raw_cache_summary_max_error=se,raw_cache_logit_max_error=ze,external_room_disjoint=True,external_office1_no_training=True,source_pretrain_source_fold_only=True,all_full_fits_cover48=True,all_external_classes0_to4=True,downloaded_selected_members=len(selection['members']),downloaded_selected_bytes=total,member_CRC_and_SHA256_verified=True,whole_archive_MD5_verified=False,old_model_hashes_verified=len(old),prior034_score_and_file_preserved=True,fits=verified,primary=primary)
    save(WORK/'verification.json',result);save(ROOT/'checks/data_revision_20261021.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
