"""Back up and extend teammate documents without altering weights or reported seed statistics."""
import csv,json,shutil,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'2026-10-25_037配方与种子复现'
BACK=P/'原始文档备份_补充前'

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(p,rr,fields):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)

def main():
    BACK.mkdir(exist_ok=True)
    for name in ['README.md','base_037.yaml','seeds_26_full_data.csv','platform_scores.csv']:
        if not (BACK/name).exists():shutil.copy2(P/name,BACK/name)
    recipe=(BACK/'base_037.yaml').read_text(encoding='utf-8-sig')
    recipe=recipe.replace("# Xiaoquan's 0.37 recipe (src/train.py --method baseline), unchanged.",
        '# Explicit hyperparameter recipe; original run source is in ../handoff/run037_20261025/original_source/.\n'
        '# Current src/train.py is a consolidated implementation, not an exact historical snapshot.')
    recipe+='''\n# 原始run补充（2026-10-25）：seed=42；unknown按identifier作采样组。\n# 每轮有放回抽960次，不保证每窗每轮都被抽到。原run无scheduler/label smoothing。\n# forward内增广：幅度/相位Gaussian std=.02；5%频率mask；\n# static前两块MixStyle Beta(.2,.2)；25% route dropout；\n# shared Dropout(.4)，temporal FFN Dropout(.1)。不能解释为“无增广”。\n# 损失：direct NLL + .15 expert CE + .5 conditional ordinal BCE\n#       + .1 JS + .1 signal-coherence proxy MSE。\n# 只训练官方48条标注，960窗，600帧/窗、stride600；不含U/外部数据。\n# 原数据清单、顺序与96个CSI/ATI的SHA：../handoff/run037_20261025/data/。\n# 原final.pt SHA256: 3519ffd3bc683044457ac8f39f8e639313a8da88dc8809da11519e03c6f0222b\n# 本包cache=std的复现是否与原缓存/窗口顺序/增广执行完全一致，仍需训练源码核对。\n'''
    (P/'base_037.yaml').write_text(recipe,encoding='utf-8')
    rr=rows(BACK/'seeds_26_full_data.csv')
    fields=list(rr[0])+['统计口径','是否独立准确率','核验备注']
    for r in rr:
        r.update(统计口径='U六房间等权；b列使用固定偏置；未重算2900窗',是否独立准确率='否',
                 核验备注='权重hash/ONNX结构/严格加载通过；good仅指TV阈值，不代表高准确率'+('；good与高判0重叠' if r['seed']=='311' else ''))
    write(P/'seeds_26_full_data.csv',rr,fields)
    incoming=rows(BACK/'platform_scores.csv')
    fields=list(incoming[0])+['sha256','文件路径','来源','记录状态']
    mapping={'原 0.37 模型（小全，best_037）':'models/best_037/model.onnx',
      '01_original_one_guard':'models/original_one_guard_20261024/01_original_one_guard.onnx',
      'gprime_full_group_fallback':'models/teammate054_20261025/gprime_full_group_fallback_web054.onnx'}
    ours=rows(ROOT/'models/platform_scores.csv');seen=set()
    for r in incoming:
        r.update(sha256='',文件路径='',来源='队友原始成绩表；非本轮重新评测',记录状态='已报告平台分')
        match=next((x for x in ours if x['path'] and x['path']==mapping.get(r['模型'])),None)
        if r['模型']=='0.49 版':
            match=next((x for x in ours if x['platform_score']=='0.49' and 'final_bias' in x['path']),None)
        if match:
            r.update(sha256=match['sha256'],文件路径=match['path']);seen.add(match['sha256'])
        if r['模型'] in ['01_original_one_guard','0.49 版']:
            r['说明']=r['说明'].replace('（队友）','（小全侧实现；双方接力）')
    for r in ours:
        if not r['platform_score'] or r['sha256'] in seen:continue
        incoming.append(dict(zip(fields,['']*len(fields))))
        incoming[-1].update({'时间(巴黎)':'历史记录/截至10-25','提交号':r.get('submission_id',''),
          '模型':r['model'],'平台分':r['platform_score'],'说明':'小全本地逐文件记录补充；同分不等于同模型',
          'sha256':r['sha256'],'文件路径':r['path'] or r.get('original_path',''),
          '来源':r['source'],'记录状态':r['status']})
        # Header uses the full-width original Chinese name.
        incoming[-1]['时间（巴黎）']=incoming[-1].pop('时间(巴黎)') if '时间（巴黎）' in fields else incoming[-1].get('时间（巴黎）','')
        incoming[-1]={k:v for k,v in incoming[-1].items() if k in fields}
        seen.add(r['sha256'])
    # Set date by the actual first field; avoid changing the received schema.
    for r in incoming:
        if not r.get(fields[0]):r[fields[0]]='历史记录/截至10-25'
    write(P/'platform_scores.csv',incoming,fields)
    shutil.copy2(ROOT/'docs/TEAM_PROGRESS_FULL_20261025.md',P/'团队完整总结_含小全实验.md')
    original=P/'original_run'
    if not original.exists():shutil.copytree(ROOT/'handoff/run037_20261025',original)
    checkpoint=original/'final.pt'
    if not checkpoint.exists():shutil.copy2(ROOT/'models/best_037/final.pt',checkpoint)
    with checkpoint.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='3519ffd3bc683044457ac8f39f8e639313a8da88dc8809da11519e03c6f0222b'
    (original/'入口说明.md').write_text('本目录final.pt是原始0.37真实检查点，非26种子中的新seed42。配方和精确数据顺序见README.md、provenance.json及data/；源码见original_source/。README中的项目相对路径对应完整小全项目，独立分享本目录时以这里的final.pt为准。原始CSI/ATI和历史大特征缓存不包含在此。\n',encoding='utf-8')
    (P/'README.md').write_text('''# 原0.37配方、种子复现与团队完整经验

2026-10-25，小全补充并实际核验。队友原始四份文档保存在 `原始文档备份_补充前/`，权重及weights_manifest保持原样。

## 先看这六点

1. 团队平台最高为0.54，尚未达到0.60。高分主线起点是小全10-19的0.37权重；队友做U直接匹配偏置到0.44、视角融合到0.47；小全做原视角1人优先到0.48、1人偏置+0.35到0.49；队友扩展完整群fallback到0.54。小全的选择性后缀微调、最新偏置版本也各0.54。
2. 小全独立结构探索也有0.38、两份0.39和融合0.40，不能概括所有得分模型都只是同一原权重。扩宽约98MB、SAM、外部数据和密集滑窗没有稳定超过主线；详见团队完整总结。
3. 原配方显式参数在base_037.yaml。原run seed42、48条标注、960不重叠窗、五轮AdamW、人数×房间有放回采样。**实际有forward内增广**，已补注；当前整理训练脚本不是原run逐字节快照。
4. 26种子包的52个权重已全部核验大小/SHA256；26份ONNX结构检查、26份PyTorch严格加载通过。seed0、42各10个真实输入窗口，两格式预测均一致。历史提交文件没包含在本包，计算与历史上传完全相同仍按原作者说明。
5. 种子42平台0.41，种子0为0.25。11份“good”仅指U预测分布TV≤.25，10份“collapsed”仅指固定偏置后U判0比例>.40；seed311同时属于两组。U没有真值，这些不是准确率。“主要是好种子运气”是待检验解释，不是已证明结论。
6. 简单集成已有失败对照：seed42+304为0.41；原权重+新seed42校正融合0.43，低于原权重单独校正0.44。下一轮先检验有条件的跨权重互补，不能只把更多seed平均。

## 文件

- `团队完整总结_含小全实验.md`：完整团队经历、成功/失败、归属与成绩；项目内原件见 `../docs/TEAM_PROGRESS_FULL_20261025.md`。
- `base_037.yaml`：原显式超参数不变，补充原run真实增广/损失/数据证据注释。
- `seeds_26_full_data.csv`：原数值不变，增加统计口径与核验说明。raw为无偏置；b为原固定偏置(-.234,-.727,+.372,+.372,+.218)。六U房间d1/h1/i1/j1/u1/v1等权，2900窗；TV=0.5*Σ|p-.2|。
- `platform_scores.csv`：保留原表并补入小全逐文件历史成绩，附SHA/路径/来源。不同文件同分保留为不同记录，不能把行数当作独立试验数量。
- `weights/`：26份原始输出ONNX与26份检查点。原manifest不改。ONNX接口frames=(600,1,3,1024,2)、intervals=(600,)，logits=(5,)，不含校准偏置。
- `original_run/`：新增原始0.37的真实final.pt、原源码快照、精确数据顺序及96文件SHA；与26个新种子分开保存，内有入口说明。
- `../docs/TEAMMATE_SEEDS_AUDIT_20261025.md`：核验、负面结果、复现差别与下一步方案。

## 原run缺的信息已经补齐

`../handoff/run037_20261025/README.md`、`provenance.json`、`data/`和`original_source/`提供seed、精确48条记录/960窗顺序、96个原始文件SHA、原始源码快照、隐含增广说明。

真实原权重已复制至本包 `original_run/final.pt`，SHA256 `3519ffd3bc683044457ac8f39f8e639313a8da88dc8809da11519e03c6f0222b`。

可分享交接zip位于 `D:/huawei2026-experiments/teammate_handoff_20261025/original_run_and_provenance.zip`，包含原final.pt；不含14GB原始数据及历史大缓存。26种子复现包没有训练源码/缓存哈希/环境，暂不能声称除了seed其他逐项相同。

## PyTorch加载

在项目根目录以本项目src/model.py的StructuredModel(size="spectral", mode="physical")构造模型，torch.load(..., weights_only=True)后严格加载checkpoint["model_state"]，调用eval()。使用torch推理时仍有预处理浮点差异，实际提交应核对ONNX输出。

截至本次补充，所有平台分均为历史成员报告；本轮没有重新调用老师测试机。
''',encoding='utf-8')
    print('UPDATED teammate documents; backed up originals; unchanged seed values and weights; score rows',len(incoming))

if __name__=='__main__':main()
