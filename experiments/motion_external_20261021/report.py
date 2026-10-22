"""Collect completed experiment evidence, register candidates, preserve platform history."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import torch
from run import ROOT,WORK,FOLDS,summary,read,save,load_data

LABELS={'baseline':'原宽度基准','capacity_control':'附加容量对照','static':'静态附加分支',
        'slow':'慢变化分支','fast':'快变化分支','three':'静态＋慢＋快三分支',
        'spectrum':'时间感知多尺度频谱','three_spectrum':'三分支＋频谱',
        'external_transfer':'Brescia单天线预训练迁移','source_pretrain_control':'官方源数据预训练／计算预算对照',
        'ratio':'天线幅度比补充分支'}

def main():
    assert (WORK/'execution_complete.json').exists(),'Do not label pending experiments complete'
    results=[read(p) for p in sorted((WORK/'fits').glob('*/metrics.json'))]
    pairs=sorted({(r['config']['variant'],r['config']['seed']) for r in results if r['validation'] is not None})
    ss=[summary(v,s) for v,s in pairs]
    paired=[]
    for variant in sorted({r['variant'] for r in ss}):
        if variant=='baseline':continue
        changes=[]
        for r in [x for x in ss if x['variant']==variant]:
            base=next(x for x in ss if x['variant']=='baseline' and x['seed']==r['seed'])
            changes.append(dict(seed=r['seed'],room_mean_delta=r['room_mean_accuracy']-base['room_mean_accuracy'],
                                macro_recall_delta=r['macro_recall']-base['macro_recall']))
        paired.append(dict(variant=variant,paired_seed_deltas=changes,
                           mean_room_delta=sum(x['room_mean_delta'] for x in changes)/len(changes),
                           improves_both_metrics_all_tested_seeds=all(x['room_mean_delta']>0 and x['macro_recall_delta']>0 for x in changes),
                           tested_seeds=len(changes)))
    delta_table='\n'.join(f"|{LABELS.get(v['variant'],v['variant'])}|{v['tested_seeds']}|{v['mean_room_delta']*100:+.2f}个百分点|" for v in paired)
    table='\n'.join(f"|{LABELS.get(r['variant'],r['variant'])}|{r['seed']}|{r['room_mean_accuracy']:.1%}|{r['macro_recall']:.1%}|{r['worst_room_accuracy']:.1%}|" for r in ss)
    audit=read(WORK/'external/cleaning_audit_v2.json');captures=audit['captures']
    kept=[r for r in captures if not r.get('excluded')]
    # Earlier audit field stored the configuration value, not a measurement of
    # available exported cores. Correct that label in this derived report only.
    for r in kept:
        if 'measured_receiving_antennas' in r:
            r['configured_receiving_antennas']=r.pop('measured_receiving_antennas')
        r['selected_actual_antenna']='core1'
    save(WORK/'external/cleaning_audit_report.json',audit)
    raw=WORK/'external/extracted'
    mat_paths=[p for p in raw.rglob('*.mat') if '__MACOSX' not in p.parts and not p.name.startswith('._')]
    raw_bytes=sum(p.stat().st_size for p in mat_paths)
    ext_counts=audit['window_counts']
    pretraining=read(WORK/'pretrain/external/metrics.json')
    ext_validation=pretraining['validation']
    majority=max(ext_counts['external_holdout'].values())/sum(ext_counts['external_holdout'].values())
    d=load_data();groups={}
    for r in d['rows']:
        domain=r['identifier'] if r['room']=='unknown' else r['room']
        key=(int(r['headcount']),domain);groups[key]=groups.get(key,0)+1
    counts={c:len({g for label,g in groups if label==c}) for c in range(5)}
    weights=[]
    for r in d['rows']:
        domain=r['identifier'] if r['room']=='unknown' else r['room'];c=int(r['headcount'])
        weights.append(1/(groups[(c,domain)]*counts[c]))
    sampler=torch.utils.data.WeightedRandomSampler(weights,960,generator=torch.Generator().manual_seed(42))
    seen=set()
    for _ in range(5):seen.update(list(sampler))
    coverage=dict(records=len({d['rows'][i]['identifier'] for i in seen}),distinct_windows=len(seen),
                  pool_windows=960,draws=4800,method='Reconstructed independent seeded sampler, same every full fit')
    for r in results:
        if r['validation'] is None:assert r['distinct_train_windows']==coverage['distinct_windows']
    candidates=[]
    for p in sorted((ROOT/'models/motion_external_20261021').glob('*.json')):
        if p.name=='registry.json':continue
        v=read(p)
        if 'sha256' not in v:continue
        v['development_results']=[r for r in ss if r['variant']==v['variant']]
        v['baseline_results']=[r for r in ss if r['variant']=='baseline']
        save(p,v);candidates.append(v)
    candidate_table='\n'.join(f"|[{v['variant']}_seed42.onnx](../{v['path']})|{v['bytes']/1e6:.2f}MB|{max(c['cpu_mean_s'] for c in v['resource']['cases']):.3f}s|未知|" for v in candidates)
    # Check the previous49 model identities; this manifest predates the current batch.
    old=read(ROOT/'checks/restarts_cleanup_20261020.json')['model_hashes']
    for name,expected in old.items():
        with (ROOT/name).open('rb') as f:value=hashlib.file_digest(f,'sha256').hexdigest()
        assert value==expected,name
    identity=read(ROOT/'submissions/next_original_seed42/identity.json')
    with (ROOT/'submissions/next_original_seed42/model.onnx').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==identity['sha256']
    snapshot=WORK/'final_source_snapshot';hashes={}
    for p in Path(__file__).parent.iterdir():
        if not p.is_file() or p.suffix not in ['.py','.md']:continue
        target=snapshot/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        with p.open('rb') as f:hashes[p.name]=hashlib.file_digest(f,'sha256').hexdigest()
    save(snapshot/'manifest.json',hashes)
    shutil.copy2(ROOT/'checks/review_20261020/external_zenodo_README.txt',WORK/'external/README_source.txt')
    shutil.copy2(Path(__file__).parent/'EXTERNAL_DATA_NOTES.md',WORK/'external/ATTRIBUTION_AND_TRANSFORMS.md')
    metadata=dict(status='complete',platform_score=None,previous_platform_best=.37,
                  best_observed_platform_unchanged=True,candidates=candidates,
                  experiment_directory=str(WORK),development_results=ss,
                  paired_comparisons=paired,
                  fits=len(results),development_fits=sum(r['validation'] is not None for r in results),
                  full_fits=sum(r['validation'] is None for r in results),
                  external_pretrain_fits=len(list((WORK/'pretrain').glob('*/metrics.json'))),
                  old_model_hashes_verified=len(old),external_window_counts=ext_counts,
                  external_mat_files=len(mat_paths),external_mat_bytes=raw_bytes)
    metadata['full_fit_coverage']=coverage
    metadata['external_pretraining']=pretraining
    save(ROOT/'models/motion_external_20261021/registry.json',metadata)
    save(ROOT/'checks/motion_external_20261021.json',metadata)
    registry=read(ROOT/'models/registry.json');registry['motion_external_20261021']=metadata
    save(ROOT/'models/registry.json',registry)
    path=ROOT/'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for v in candidates:
        if not any(r['sha256']==v['sha256'] for r in rows):
            rows.append(dict(model=LABELS.get(v['variant'],v['variant'])+' seed42',platform_score='',path=v['path'],
                             sha256=v['sha256'],source='local research; platform accuracy unknown',
                             status='research_only_external_terms_unverified' if v['external_data'] else 'awaiting_platform',submission_id=''))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=f'''# 2026-10-21：运动证据与Brescia外部数据实验

本轮已完成。平台已确认最高仍为两份0.37，新文件均没有平台成绩；模型结构名称不等于已经检测出静止／慢动／快动人体。实验计划在训练前登记于 experiments/motion_external_20261021/PLAN.md，附加幅度比对照见RATIO_PLAN.md。

## 外部数据实际核查

来源：[Zenodo17225694 v2](https://zenodo.org/records/17225694)，CC BY-SA4.0。只取0—4人MAT和配置，跳过PCAP、5人和macOS附属文件。已取得{len(mat_paths)}个实际MAT，解压后合计{raw_bytes/1e9:.2f}GB；ZIP成员下载时校验CRC，单天线清洗记录另存SHA256。未声称下载并校验完整ZIP的MD5。

数据并非全部30Hz：有约30、100、200Hz采集，实际以MAT时间戳为准。3／4人旧记录为单Rx，且仅有CSI矩阵，无实测时间戳，时间按配置标为nominal_only。作者提到中心子载波异常，外部清洗屏蔽MATLAB500—524位置；官方输入的无效频点模式并不完全相同，不宣称硬件格式完全等价。

3／4人记录为坐着使用电脑，不能当作严格静止人体、慢走和快走三类动作标签。只有一个办公室、不同设备位置／日期，人数与采集条件仍然耦合；很多配置注明走廊有人经过，与题目条件存在差异。它增加外部环境观察，不能直接宣称补齐跨房间泛化。

清洗后使用真实core1单天线的幅度，裁取约3秒窗口并最多保留每次采集20个窗口。150点重采样是明确标注的插值，不是补出真实200Hz运动观测。对旧无时间戳记录仅是名义3秒。使用源文件SHA去重，并以官方有标签数据每100帧、所有Rx的17280个精确内容锚点扫描外部core1帧；该检查不排除变换后的副本或未覆盖重合。

外部训练窗口按0—4类：{ext_counts['external_train']}；日期隔离外部验证：{ext_counts['external_holdout']}。外部验证不覆盖完整0—4类，因此不能报作独立五类计数准确率。完整会话不跨训练与验证，官方房间验证的分母未加入外部样本。单天线清洗采用{len(kept)}条采集，审计保留活动、设备位置、时间来源、间隔和排除原因。

## 如何接入

不复制天线，不把3／4人单天线伪装成完整3Rx输入。先训练单天线幅度编码器，将兼容的卷积／投影参数迁移到原模型静态路线；首层单通道权重扩到三个输入并除以3，其余路线重新初始化。然后按相同5轮官方监督预算微调。

同时运行仅使用各折官方源数据的单天线预训练对照，匹配每轮更新数，以区分额外计算与外部数据来源的影响。开发模型没有加载见过全部官方标签的0.37权重；未用官方150条U。全量候选才使用全部48条有标签记录。

外部预训练末轮损失{pretraining['history'][-1]['loss']:.4f}，但独立日期的0／1人样本上准确率仅{ext_validation['accuracy']:.1%}；在该留出上恒预测最多类别的参考为{majority:.1%}。这不是完整五类评测，也不是平台成绩；低训练损失没有迁移成日期泛化，不能把它当作有效人数教师。是否帮助官方任务仍由下面的迁移对照判断。

## 官方整房间开发结果

|方案|初始化种子|三房间均值|合并五类召回|最差房间|
|---|---:|---:|---:|---:|
{table}

这是反复使用过的a1/c1/g1开发集，不是新盲测。先按seed42筛选，再对前两结构用2026复核，存在筛选偏差。平均与最差房间并列报告；训练分数不作为隐藏准确率。容量对照只与静态附加分支匹配参数量，不是每个组合的精确等参数对照。频谱分支是时间感知幅度变化谱，不是校准速度图，也不按峰数数人。IAA等高成本自适应谱估计未运行，不宣称穷尽所有可能算法。

|方案|已测种子数|相对于同种子基准的房间均值差，跨已测种子平均|
|---|---:|---:|
{delta_table}

单种子改善不构成稳定提升。外部迁移还应与source_pretrain_control比较，不能只与未增加预训练计算的基准比较；逐项差值和混淆矩阵保存在登记JSON。

## 保留模型与部署检查

|文件|大小|本机最慢输入平均CPU时间|平台|
|---|---:|---:|---|
{candidate_table}

全部列出文件通过独立CPU进程接口、内嵌权重、数值一致性、<100000000字节、平均<0.7秒及内存<6GB检查。本机速度不是老师机器保证。全48训练文件没有独立本地准确率；表中的开发准确率来自不同的折模型。

完成{metadata['development_fits']}次官方房间开发拟合、{metadata['full_fits']}次全量拟合，以及{metadata['external_pretrain_fits']}次单天线预训练。所有失败／较差结果保留。旧49份ONNX哈希未改变，两份0.37身份和成绩保持原样。

全量有放回采样重放核查：实际覆盖{coverage['records']}条记录、{coverage['distinct_windows']}个不同窗口，共4800次抽样；入池960窗口不表示每个都被抽到。

## 许可与其他数据

保留作者、版本、许可和变换说明。不能根据“不再分发原始数据”就断定模型权重没有相同方式共享义务，也不能反向断定所有权重必然受SA约束；[CC官方FAQ](https://creativecommons.org/faq/#artificial-intelligence-and-cc-licenses)没有在此替我们判定具体权重是否构成改编作品。外部训练权重先标研究用途，未代用户提交平台，也未发送任何对外消息。赛事外部数据规则及发布条件仍待确认。

其他候选详见 [外部核查笔记](../experiments/motion_external_20261021/EXTERNAL_DATA_NOTES.md)：RadioPoints三房间0—7人适合后续多房间迁移，但硬件不同且README非商业条款需核对；80MHz802.11ac数据环境更多但频点不等价；Exposing the CSI有实际视频和活动标签但不是完整0—4人数集。本轮没有额外下载这些大数据集。

原始MAT、清洗特征、逐窗口清单、权重、逐折预测和源码位于 `{WORK}`。旧平台最佳保留；后续提交由用户取得平台反馈，不能用开发结果宣布超过0.37。
'''
    (ROOT/'docs/MOTION_EXTERNAL_20261021.md').write_text(report,encoding='utf-8')
    (ROOT/'models/motion_external_20261021/README.md').write_text('# 本轮研究模型\n\n平台成绩均未知，历史最高仍0.37。慢／快分支本地复核未超过基准，不优先推荐替换0.37。外部迁移未优于基准，未导出全量提交模型。\n\n完整报告：[MOTION_EXTERNAL_20261021.md](../../docs/MOTION_EXTERNAL_20261021.md)。\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['candidates','development_results']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
