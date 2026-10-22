from common import ROOT,WORK,FeatureStore,write_csv,save
from engine import ordered_indices

store=FeatureStore();summary=[]
for fold in ('a1','c1','g1','e1'):
    folder=WORK/'splits'/f'leave_{fold}'
    tests={name:ordered_indices(store,recipe,fold,False) for name,recipe in (('test_original','baseline'),('test_dense','step_0p5s'))}
    for name,ix in tests.items():write_csv(folder/f'{name}.csv',[store.rows[i] for i in ix])
    for recipe in ('step_1s','step_0p5s'):
        ix=ordered_indices(store,recipe,fold,True);train_ids={store.rows[i]['identifier'] for i in ix};test_ids={store.rows[i]['identifier'] for i in tests['test_dense']};assert train_ids.isdisjoint(test_ids)
        write_csv(folder/f'train_{recipe}.csv',[store.rows[i] for i in ix]);summary.append(dict(fold=fold,recipe=recipe,training_windows=len(ix),training_records=len(train_ids),test_original_windows=len(tests['test_original']),test_dense_windows=len(tests['test_dense']),test_records=len(test_ids),record_disjoint=True))
for recipe in ('step_1s','step_0p5s'):
    write_csv(WORK/'splits/full'/f'train_{recipe}.csv',[store.rows[i] for i in ordered_indices(store,recipe,'full',True)])
save(WORK/'splits/summary.json',summary);save(ROOT/'checks/sliding_splits_20261022.json',summary);print(summary)
