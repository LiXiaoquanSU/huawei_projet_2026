import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
files=list((ROOT/'models/spectral_diversity_20261022').glob('01_*.json'));assert len(files)==1
m=json.loads(files[0].read_text(encoding='utf-8'));assert m['platform_score'] is None
for name in ('README.md','models/README.md','submissions/README.md','docs/SUMMARY.md','docs/EXPERIMENT_INDEX.md'):
    p=ROOT/name;text=p.read_text(encoding='utf-8');prefix='' if name=='README.md' else '../'
    paragraphs=text.split('\n\n')
    paragraphs=[q for q in paragraphs if not q.startswith('**最新空间结构研究（2026-10-22）：**')]
    text='\n\n'.join(paragraphs)
    report='SPECTRAL_DIVERSITY_20261022.md' if name.startswith('docs/') else prefix+'docs/SPECTRAL_DIVERSITY_20261022.md'
    note=f"**谱多样性结构研究（2026-10-22）：** 两阶段33次拟合完成。新候选[{m['architecture']} seed{m['seed']}]({prefix}{m['path']})，{m['bytes']/1e6:.2f}MB；两种子三房间均值{m['development']['two_seed_room_mean']:.2%}，超过原对照指标：{m['improved_development']}，平台未知。两份原0.39仍是平台最佳，上一金字塔版已测0.37。[实验报告]({report})。"
    if note not in text:
        title,body=text.split('\n',1);text=title+'\n\n'+note+'\n'+body
    p.write_text(text,encoding='utf-8')
print('Current indexes updated.')
