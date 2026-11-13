"""Record completed research without changing any reported platform score."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
status = ('本轮已完成环境双验证研究：48次小模块训练、24次天线关系图训练和环境子空间对照。'
          '图模型最好在留房间代理上净增6/880窗，未通过候选门槛；没有新的推荐ONNX。'
          '最高已测仍0.54，0.60尚未达到。'
          '诊断发现代理602个错误中402个被固定保护分支锁定，后续应单独研究可靠的条件解锁，不能直接取消全部保护。'
          '完整结果见docs/ENVIRONMENT_PROTOCOL_20261025.md。')

for name in ['README.md', 'docs/TEAM_PROGRESS_FULL_20261025.md']:
    path = ROOT / name
    text = path.read_text(encoding='utf-8')
    if status not in text:
        heading, rest = text.split('\n', 1)
        path.write_text(heading + '\n\n' + status + '\n\n' + rest.lstrip('\n'), encoding='utf-8')

registry_path = ROOT / 'models/registry.json'
registry = json.loads(registry_path.read_text(encoding='utf-8'))
assert registry['current_best']['platform_score'] == .54
assert registry['next_submission'] is None
registry['research_status'] = 'environment_dual_protocol_72fits_no_qualified_candidate'
registry['research_report'] = 'docs/ENVIRONMENT_PROTOCOL_20261025.md'
registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

package = ROOT / '2026-10-25_037配方与种子复现'
(package / '团队完整总结_含小全实验.md').write_text((ROOT / 'docs/TEAM_PROGRESS_FULL_20261025.md').read_text(encoding='utf-8'), encoding='utf-8')
roadmap = ROOT / 'docs/TEST_STRUCTURE_AND_ROADMAP_20261025.md'
text = roadmap.read_text(encoding='utf-8')
note = '\n\n## 路线执行补记\n\n' + status + '\n'
if '## 路线执行补记' not in text:
    roadmap.write_text(text + note, encoding='utf-8')
(package / '隐藏测试结构与下一阶段路线.md').write_text(roadmap.read_text(encoding='utf-8'), encoding='utf-8')
print(status)
