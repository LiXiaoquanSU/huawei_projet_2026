"""Replace a quality-failed class4 recording without consulting any model score."""
import json
from pathlib import Path
ROOT=Path('D:/huawei2026-experiments/data_revision_20261021/ehunam')
if (ROOT/'supplement.json').exists():print('SUPPLEMENT_ALREADY_REGISTERED');raise SystemExit(0)
p=ROOT/'selection.json';s=json.loads(p.read_text());allrows=json.loads((ROOT/'summary_metadata.json').read_text())
candidates=[r for r in allrows if r['Set']=='MC1-05' and r['N_People']==4 and r['Rx']==1 and r['Application']=='PC' and r['File'] not in s['members']]
chosen=max(candidates,key=lambda r:r['N_CSI'])
if not (ROOT/'initial_selection.json').exists():(ROOT/'initial_selection.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
s['members'].append(chosen['File']);s['records'].append(chosen);s['supplement_reason']='One selectedClassroom4-person capture has no valid3s window after5s edge trimming. Add highestN_CSI unselectedClassroom4-personRx1 record, without inspecting model scores.'
index=json.loads((ROOT/'archive_index.json').read_text());sizes={r['name']:r['bytes'] for r in index['members']};s['bytes']=sum(sizes[n] for n in s['members'])
p.write_text(json.dumps(s,indent=2),encoding='utf-8');(ROOT/'supplement.json').write_text(json.dumps(dict(members=[chosen['File']],reason=s['supplement_reason']),indent=2),encoding='utf-8');print(chosen['File'],s['bytes'])
