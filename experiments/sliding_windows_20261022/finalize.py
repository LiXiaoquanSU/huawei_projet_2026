import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/sliding_windows_20261022')
metas=json.loads((WORK/'exports.json').read_text(encoding='utf-8'));top=metas[0]
counts=json.loads((WORK/'cache/index.json').read_text(encoding='utf-8'))['counts']
for name in ('README.md','models/README.md','submissions/README.md','docs/SUMMARY.md','docs/EXPERIMENT_INDEX.md'):
    p=ROOT/name;text=p.read_text(encoding='utf-8');prefix='' if name=='README.md' else '../'
    report='SLIDING_WINDOWS_20261022.md' if name.startswith('docs/') else prefix+'docs/SLIDING_WINDOWS_20261022.md'
    note=f"**重叠窗口数据实验（2026-10-22）：** 暂停新结构探索；原960窗扩为1秒起窗{counts['step_1s']}窗、0.5秒起窗{counts['step_0p5s']}窗，独立记录仍48。33次拟合完成，三既有结构重训并导出。当前研究候选[{top['architecture']} {top['recipe']}]({prefix}{top['path']})，主测试均值{top['development']['two_seed_room_mean']:.2%}，平台未知，不继承旧分数。原0.39仍是平台最佳。[数据与训练报告]({report})。"
    if note not in text:
        title,body=text.split('\n',1);text=title+'\n\n'+note+'\n'+body
    p.write_text(text,encoding='utf-8')
report=ROOT/'docs/SLIDING_WINDOWS_20261022.md';text=report.read_text(encoding='utf-8')
if '## 各模型交付与限制' not in text:
    text+='\n## 各模型交付与限制\n\n'
    for m in metas:
        d=m['development'];training=m['training'];cm=[r['confusion_matrix'] for r in d['results']];support=[sum(sum(matrix[k]) for matrix in cm) for k in range(5)];recalls=[sum(matrix[k][k] for matrix in cm)/support[k] for k in range(5)]
        text+=f"- {m['architecture']} {m['recipe']} seed{m['seed']}：{m['path']}；{m['bytes']/1e6:.2f}MB，SHA {m['sha256']}。主测试{d['baseline_room_mean']:.2%}→{d['two_seed_room_mean']:.2%}，预设指标改善{d['improved_development']}；0—4人召回{recalls}；e1 {d['e1']['accuracy']:.2%}。全量抽样实际覆盖{training['actual_windows']}/{training['config']['training_windows']}新增池窗口，48条记录均参与。CPU最慢用例均值{max(c['cpu_mean_s'] for c in m['resource']['cases']):.4f}s。\n"
        a=m['sampling_audit'];text+=f"  原始帧抽样覆盖率（各原记录等权）{a['old_full_raw_frame_coverage']:.2%}→{a['mean_raw_frame_coverage']:.2%}。窗口数增多并未扩大同预算下见过的不同原始帧数，增加的是窗口上下文组合。\n"
    text+='\n重叠窗提供更多起点，不能补齐缺失的房间×人数组合。本轮固定更新次数，新增候选池并未完全遍历；若需要增加训练预算，应另设非重叠窗口同预算对照。两种子仍不足以认定统计显著，开发集反复使用，最终改善需平台验证。\n'
    text+='\n数据清单：cache/windows.csv 为全部窗口；splits/leave_a1、leave_c1、leave_g1、leave_e1 分别含训练1秒/0.5秒CSV、test_original.csv和test_dense.csv；splits/full为全量训练池。sampling_audit.json及各fit的sampled_windows.csv记录真正抽到的窗口。\n'
report.write_text(text,encoding='utf-8');(WORK/'REPORT.md').write_text(text,encoding='utf-8')
for p in Path(__file__).parent.iterdir():
    if p.suffix in ('.py','.md'):shutil.copy2(p,WORK/'source_snapshot'/p.name)
manifest=WORK/'source_manifest.json';files=json.loads(manifest.read_text(encoding='utf-8'))
for name in ('experiments/global_local_20261021/network.py','experiments/structure_events_20261021/network.py','experiments/fresh_backbones_20261022/network.py'):
    files[name]=''
for name in files:
    with (ROOT/name).open('rb') as f:files[name]=hashlib.file_digest(f,'sha256').hexdigest()
    target=WORK/'source_snapshot/dependencies'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
manifest.write_text(json.dumps(files,indent=2),encoding='utf-8');print('Indexes and per-model findings updated.')
