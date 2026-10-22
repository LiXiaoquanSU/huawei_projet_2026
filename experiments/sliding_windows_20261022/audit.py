import csv
import numpy as np
from common import ROOT,WORK,RAW,FeatureStore,old_data,save,read,sha
from prepare import starts_from_clock

def main():
    clock=np.full(12000,5000,dtype=np.float32);clock[-1]=0
    _,base,one,half=starts_from_clock(clock)
    assert base==set(range(0,11401,600)) and one==set(range(0,11401,200)) and half==set(range(0,11401,100))
    old=old_data();store=FeatureStore();official=list(csv.DictReader((RAW/'samples.csv').open(encoding='utf-8-sig')));lookup={r['identifier']:r for r in official}
    assert len(lookup)==48 and {r['identifier'] for r in store.rows}==set(lookup)
    keys={(r['identifier'],r['start']) for r in store.rows};assert len(keys)==len(store.rows)
    oldlookup={(r['identifier'],int(r['start'])):i for i,r in enumerate(old['rows'])};anchors=0
    for i,r in enumerate(store.rows):
        o=lookup[r['identifier']];assert int(r['headcount'])==int(o['headcount']) and r['room']==o['room'] and r['layout']==o['layout']
        assert r['stop']-r['start']==600 and 0<=r['start']<=11400 and r['window_seconds']>=0
        assert not r['step_1s'] or r['step_0p5s']
        if r['baseline']:
            j=oldlookup[(r['identifier'],r['start'])];t,s=store.get(i)
            assert np.array_equal(t.numpy(),old['t'][j].numpy()) and np.array_equal(s.numpy(),old['s'][j].numpy());anchors+=1
    assert anchors==960
    cache_files=0
    for p in (WORK/'cache/shards').glob('*/record.json'):
        item=read(p);audit=item['audit'];assert sha(p.parent/'t.npy')==audit['cache_t_sha256'] and sha(p.parent/'s.npy')==audit['cache_s_sha256'];cache_files+=1
    assert cache_files==48
    for fold in ('a1','c1','g1','e1'):
        for recipe in ('step_1s','step_0p5s'):
            tr={store.rows[i]['identifier'] for i in store.indices(recipe,fold,True)};va={store.rows[i]['identifier'] for i in store.indices('step_0p5s',fold,False)};assert tr.isdisjoint(va)
    result=dict(status='pass',source_records=48,old_anchor_features_identical=anchors,unique_windows=len(keys),counts=store.index['counts'],official_labels_verified=True,record_disjoint=True,synthetic_true_clock_stride_test=True,cache_hashes_verified=48,independent_records_added=0)
    save(ROOT/'checks/sliding_integrity_20261022.json',result);save(WORK/'cache/audit.json',result);print(result)
if __name__=='__main__':main()
