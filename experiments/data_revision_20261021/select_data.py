"""Read author metadata without editing the workbook; output reproducible selection."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import openpyxl
ROOT=Path('D:/huawei2026-experiments/data_revision_20261021/ehunam')
w=openpyxl.load_workbook(ROOT/'Summary.xlsx',read_only=True,data_only=True)
v=iter(w.active.values);headers=next(v);rows=[dict(zip(headers,row)) for row in v]
def serial(r):return {k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in r.items()}
(ROOT/'summary_metadata.json').write_text(json.dumps([serial(r) for r in rows],indent=2),encoding='utf-8')
candidates=[r for r in rows if r['Application'] in ('PC','E') and r['BW']==80 and r['N_People'] in range(5) and r['N_Machine']==0 and r['Set'].startswith('MC1')]
counts=Counter((r['Set'],r['Enviroment'],r['N_People']) for r in candidates)
print('CANDIDATES',len(candidates))
for k,v in sorted(counts.items()):print(k,v)
print('ONE_RX',Counter((r['Enviroment'],r['N_People']) for r in candidates if r['Rx']==1))
print('FILES',[serial(r) for r in candidates if r['Rx']==1][:3])
# Up to two independent participant configurations per count, set and receiver.
# KeepRx1 only: these are independent receivers, NOT coherent antenna elements.
groups=defaultdict(list)
for r in candidates:
    if r['Rx']==1 and r['N_CSI']>=3000:groups[(r['Set'],r['N_People'])].append(r)
selected=[]
for key,group in sorted(groups.items()):
    group.sort(key=lambda r:(r['People'] or '',r['Number']))
    chosen=[group[0]] if len(group)==1 else [group[0],group[-1]]
    selected.extend(chosen)
index=json.loads((ROOT/'archive_index.json').read_text());sizes={r['name']:r['bytes'] for r in index['members']}
selection=dict(members=[r['File'] for r in selected],records=[serial(r) for r in selected],bytes=sum(sizes[r['File']] for r in selected),rule='MC1 PC/E BW80 count0--4 no machines; realRx1 only; at least3000 CSIs; first+last distinct author participant/number configuration per set and count; no window/random receiver split')
(ROOT/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
print('SELECTED',len(selected),'BYTES',selection['bytes'])
