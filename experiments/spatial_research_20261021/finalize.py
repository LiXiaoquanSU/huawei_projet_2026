"""Refresh current navigation without erasing historical experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
meta = json.loads((ROOT / 'models/spatial_research_20261021/01_pyramid_context_seed42.json').read_text(encoding='utf-8'))
if meta['platform_score'] is not None:
    raise SystemExit('Historical candidate already evaluated; current feedback must not be overwritten.')
for name in ('README.md', 'models/README.md', 'submissions/README.md', 'docs/SUMMARY.md', 'docs/EXPERIMENT_INDEX.md'):
    path = ROOT / name
    content = path.read_text(encoding='utf-8')
    paragraphs = content.split('\n\n')
    for i, paragraph in enumerate(paragraphs):
        if paragraph.startswith(('**下一提交研究候选：**', '下一提交研究候选：', '下一研究候选：')) and 'ordinal_readout' in paragraph:
            paragraphs[i] = '序数读出候选已测0.38，撤回优先推荐；旧文件保留用于复核。'
        elif paragraph.startswith('本轮0.39后研究已完成28次新拟合：'):
            paragraphs[i] = paragraph.split('固定序数读出')[0] + '固定序数读出后来已获平台0.38，未超过两份原0.39。'
    content = '\n\n'.join(paragraphs)
    if '最新空间结构研究（2026-10-22）' not in content:
        prefix = '' if name == 'README.md' else '../'
        report = 'SPATIAL_RESEARCH_20261021.md' if name.startswith('docs/') else prefix + 'docs/SPATIAL_RESEARCH_20261021.md'
        notice = f'''**最新空间结构研究（2026-10-22）：** 已完成18次拟合。新研究候选为[多层频率特征＋上下文事件 seed42]({prefix}models/spatial_research_20261021/01_pyramid_context_seed42.onnx)，84.97MB，平台未测。两种子三房间均值27.57%，低于原跨注意力29.24%和上下文28.66%，未达到60%；不能称为已验证升级。原两份0.39仍是平台最佳。[完整报告]({report})。统一提交副本：`submissions/spatial_research_20261021/model.onnx`。'''
        head, tail = content.split('\n', 1)
        content = head + '\n\n' + notice + '\n' + tail
    path.write_text(content, encoding='utf-8')
print('Current indexes updated; old models preserved.')
