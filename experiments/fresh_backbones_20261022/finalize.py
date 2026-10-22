import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
files=list((ROOT/'models/fresh_backbones_20261022').glob('01_*.json'));assert len(files)==1
m=json.loads(files[0].read_text(encoding='utf-8'));assert m['platform_score'] is None
for name in ('README.md','models/README.md','submissions/README.md','docs/SUMMARY.md','docs/EXPERIMENT_INDEX.md'):
    p=ROOT/name;text=p.read_text(encoding='utf-8');prefix='' if name=='README.md' else '../'
    paragraphs=text.split('\n\n')
    paragraphs=[q for q in paragraphs if not q.startswith(('**最新空间结构研究（2026-10-22）：**','**谱多样性结构研究（2026-10-22）：**'))]
    text='\n\n'.join(paragraphs)
    report='FRESH_BACKBONES_20261022.md' if name.startswith('docs/') else prefix+'docs/FRESH_BACKBONES_20261022.md'
    note=f"**独立骨干研究（2026-10-22）：** 24次神经网络＋8次随机字典拟合完成。研究对照[{m['architecture']} seed{m['seed']}]({prefix}{m['path']})，{m['bytes']/1e6:.2f}MB；两种子三房间均值{m['development']['two_seed_room_mean']:.2%}，未超过同12轮对照或历史5轮最佳，不列优先提交，平台未知。两份原0.39仍是平台最佳，上一谱多样性版已测0.35。[实验报告]({report})。"
    if note not in text:
        title,body=text.split('\n',1);text=title+'\n\n'+note+'\n'+body
    p.write_text(text,encoding='utf-8')
regfile=ROOT/'models/registry.json';reg=json.loads(regfile.read_text(encoding='utf-8'))
if reg['recommendations'].get('next_submission')=='fresh_backbones_20261022':
    reg['recommendations']['next_submission']='pending_better_candidate'
    reg['recommendations']['next_submission_note']='No verified upgrade in latest independent-backbone batch; Mixer retained as optional research control.'
    reg['recommendations']['exploratory_candidate']='fresh_backbones_20261022'
regfile.write_text(json.dumps(reg,ensure_ascii=False,indent=2),encoding='utf-8')
report_path=ROOT/'docs/FRESH_BACKBONES_20261022.md';report=report_path.read_text(encoding='utf-8')
lesson='''
## 本轮经验

1. 确实换了独立骨干，结果仍不佳：Mixer两种子21.24%、BiGRU20.13%、Patch首种子14.70%、随机字典两种子9.98%。不能据此断言这些架构永远不适合，只能说当前输入组织、训练与数据条件下失败。
2. Mixer全量训练数据重评96.15%，与留房间21.24%差距大；容量不足不是当前主要证据支持的解释。继续扩大模型或训练到99%没有泛化保证。
3. Mixer合并两种子的4人召回10.71%，但4人预测精确率仅13.33%，不是可靠突破。e1为33.33%是所有样本判1人，不能当有效识别。
4. 原上下文12轮对照也未超过其5轮结果，延长训练可能强化源环境记忆。未按验证峰值挑epoch，更没有把96.15%当测试准确率。
5. 所有方法共用缓存及幅相预处理，这次否定不了原始复数CSI上其他方案。进一步探索应明确区分“输入丢失信息”和“学到房间捷径”，优先尝试受约束的复数/相对通道表示、按源记录进行自监督预训练及背景扰动一致性，而不是再次无目标地堆容量。这些是后续假设，本轮未实施，不能赋予准确率。

交付的Mixer仅研究对照，未确认改进；本轮不指定优先替代0.39的模型。
'''
if '## 本轮经验' not in report:report+=lesson
report_path.write_text(report,encoding='utf-8')
work=Path('D:/huawei2026-experiments/fresh_backbones_20261022');(work/'REPORT.md').write_text(report,encoding='utf-8')
other=Path('D:/huawei2026-experiments/random_dictionary_20261022');(other/'REPORT.md').write_text('本阶段失败，不导出。\n\n'+report,encoding='utf-8')
for folder in (Path(__file__).parent,ROOT/'experiments/random_dictionary_20261022'):
    dest=(work if folder.name=='fresh_backbones_20261022' else other)/'source_snapshot'
    for p in folder.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,dest/p.name)
manifest=work/'source_manifest.json';files=json.loads(manifest.read_text(encoding='utf-8'))
for name in files:
    with (ROOT/name).open('rb') as f:files[name]=hashlib.file_digest(f,'sha256').hexdigest()
manifest.write_text(json.dumps(files,indent=2),encoding='utf-8')
print('Current indexes and findings updated. No priority replacement recommended.')
