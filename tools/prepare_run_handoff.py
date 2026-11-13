"""Record the latest054 and package evidence for the original run, not a new experiment."""
import ast,csv,hashlib,io,json,shutil,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'handoff/run037_20261025'
ARCHIVE=Path('D:/huawei2026-experiments/project_archive_20261023_044')
PACKAGE=Path('D:/huawei2026-experiments/teammate_handoff_20261025/original_run_and_provenance.zip')
CP=ROOT/'models/best_037/final.pt'
PAIRED=Path('D:/huawei2026-experiments/u_ssl_20261025/fits/actual_paired_s42/step120.pt')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def score():
    rel='models/robust_bias054_20261025/01_selective054_bias_stable.onnx';p=ROOT/rel;m=read(p.with_suffix('.json'));assert sha(p)==m['sha256']=='9966149c7f122254a647e7a14921d3104b595b7fc7237c7b0e506c1bf2ad83cf'
    m.update(platform_score=.54,status='scored_tied_best',source='user-reported platform0.54;2026-10-25;submissionIDunknown');save(p.with_suffix('.json'),m)
    r=read(ROOT/'models/registry.json');r['research_candidates']=[m if x.get('sha256')==m['sha256'] else x for x in r['research_candidates']]
    if (r.get('next_submission') or {}).get('sha256')==m['sha256']:r['next_submission']=None
    tied=[x for x in r.get('tied_current_best',[]) if x.get('sha256')!=m['sha256']]+[m];r['tied_current_best']=tied;r['tied_best_054']=tied;r['latest_scored']=m;r['research_status']='bias054_tied_prepare_original_run_handoff';r['candidate_note']='Repeatedrecordbias scored054, tied nothigher. No next experiment started; originalrun seed/data/source/checkpoint handoff prepared.';save(ROOT/'models/registry.json',r)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[x for x in rows if x['sha256']==m['sha256']];assert len(matches)==1;matches[0].update(platform_score='0.54',status=m['status'],source=m['source'])
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return m
def main():
    latest=score();OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True);(OUT/'original_source').mkdir(exist_ok=True);(OUT/'evidence').mkdir(exist_ok=True)
    entries={e['path']:e for e in read(ARCHIVE/'manifest.json')['entries']};prefix='_archive/history_20261019/artifacts/submission_structured_20261019/';provenance=[]
    def extract(original,target,kind):
        e=entries[original];blob=ARCHIVE/e['blob'];assert sha(blob)==e['sha256'];dest=OUT/target;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==e['sha256']
        else:shutil.copy2(blob,dest)
        provenance.append(dict(path=target,original_path=original,sha256=e['sha256'],evidence_kind=kind));return dest
    for n in ('config.json','history.csv'):extract(prefix+n,'evidence/original_'+n,'original_submission_artifact')
    names=('refit_structured.py','train_structured.py','structured_model.py','export_structured.py','validate_model.py')
    for n in names:extract(prefix+'source_snapshot/'+n,'original_source/'+n,'original_submission_source_snapshot')
    queue=list(names);done=set()
    while queue:
        n=queue.pop(0)
        if n in done:continue
        done.add(n);tree=ast.parse((OUT/'original_source'/n).read_text(encoding='utf-8-sig'))
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules += [v.name.split('.')[0] for v in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:modules.append(node.module.split('.')[0])
        for mod in sorted(set(modules)):
            fn=mod+'.py';original='_archive/history_20261019/'+fn
            if fn not in done and not (OUT/'original_source'/fn).exists() and original in entries:extract(original,'original_source/'+fn,'historical_dependency_archive_not_claimed_exact_run_snapshot');queue.append(fn)
    originalrows=[]
    for split in ('train','val'):
        p=extract('_archive/history_20261019/artifacts/baseline_a1_seed42/'+split+'_windows.csv','evidence/original_bank_'+split+'_windows.csv','original_feature_bank_record_order')
        with p.open(encoding='utf-8-sig',newline='') as f:originalrows+=list(csv.DictReader(f))
    samples=Path('D:/huawei2026-data/labeled/samples.csv')
    with samples.open(encoding='utf-8-sig',newline='') as f:sample_rows=list(csv.DictReader(f))
    originalrows += [dict(r,start=i,stop=i+600) for r in sample_rows if r['room']=='unknown' for i in range(0,12000,600)]
    manifest=ROOT/'manifests/train_windows.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    key=lambda r:(r['identifier'],int(r['start']),int(r['stop']),int(r['headcount']))
    assert list(map(key,originalrows))==list(map(key,rows)) and len(rows)==960
    shutil.copy2(manifest,OUT/'data/train_windows.csv');shutil.copy2(samples,OUT/'data/samples.csv')
    prior={r['identifier']:r for r in read(Path('D:/huawei2026-experiments/actual047_audit_20261024/summary.json'))['source_files']};records=[]
    identifiers=list(dict.fromkeys(r['identifier'] for r in rows));assert len(identifiers)==48
    for j,rid in enumerate(identifiers):
        rr=[r for r in rows if r['identifier']==rid];r=rr[0];record=dict(identifier=rid,layout=r['layout'],room=r['room'],headcount=int(r['headcount']),windows=len(rr),window_frames=600,stride_frames=600)
        for part in ('CSI','ATI'):
            p=Path('D:/huawei2026-data/labeled')/f'{part}_{rid}.b2';h=sha(p);assert h==prior[rid][part.lower()+'_sha256'];record[part.lower()+'_file']=p.name;record[part.lower()+'_bytes']=p.stat().st_size;record[part.lower()+'_sha256']=h
        records.append(record)
        if (j+1)%12==0:print('HANDOFF_DATA_HASHED',j+1,flush=True)
    csvwrite(OUT/'data/records48_sha256.csv',records)
    cp=torch.load(CP,weights_only=True,map_location='cpu');assert sha(CP)=='3519ffd3bc683044457ac8f39f8e639313a8da88dc8809da11519e03c6f0222b';assert cp['config']==read(OUT/'evidence/original_config.json')
    ext=torch.load(PAIRED,weights_only=True,map_location='cpu');assert ext['seed']==42 and ext['step']==120 and ext['checkpoint_sha256']==sha(CP)
    completion=read(PAIRED.parent/'complete.json');save(OUT/'evidence/paired_seed42_complete.json',completion)
    facts=dict(original_checkpoint=dict(path=str(CP),sha256=sha(CP),bytes=CP.stat().st_size,keys=list(cp),config=cp['config'],original_archive_path=prefix+'final.pt'),original_manifest=dict(path='data/train_windows.csv',sha256=sha(manifest),records=48,windows=960,exact_order_reconstructed_from_original_bank=True),raw_data=dict(root='D:/huawei2026-data/labeled',freshly_hashed_files=96,all_match_previous_audit=True),source_provenance=provenance,paired_extension=dict(path=str(PAIRED),sha256=sha(PAIRED),bytes=PAIRED.stat().st_size,seed=42,step=120,used_interpolation=.25,not_a_complete_original_final_pt=True),current_submission=dict(path=latest['path'],sha256=latest['sha256'],platform_score=.54,final_delta=latest['delta']),limitations=['Originalfinal.pt storesmodel_state/config only,nooptimizer/RNGstate.','Current src/train.py isconsolidatedlater,NOToriginalrunexactscript.','OriginalcachedTF/phase/basefeaturebanksnotbundled; rawrecomputationnotclaimedbitidentical.','Dependencyarchiveslabelledhistorical,notallprovenexactrun snapshots.','Original037final.pt alone doesnotreproduce054TTArouting/FP16storage/pairedsuffix/finalbias.'])
    save(OUT/'provenance.json',facts)
    readme='''# 原run核对与队友交接

主文件 final.pt 是原始0.37底座，不是新训练出来的替代品。其配置直接从checkpoint及原提交归档交叉确认。

- 随机种子42：Python/NumPy/Torch/CUDA及WeightedRandomSampler均设42；spectral/physical，5个epoch，来自a1/c1/g1开发最佳epoch[1,19,5]的中位数。全量训练重新随机初始化，不是把三个fold权重平均。
- 比赛有标签48条记录全部训练，960个600帧不重叠窗口，stride600；包含3条room=unknown但人数标签为1的记录。每条20窗。没有外部数据、官方U、重叠滑窗或人为动作/房间面积标签用于这个原run。
- data/records48_sha256.csv给出全部96个CSI/ATI文件名、标签、大小和本次重新核验SHA256；data/train_windows.csv给出960窗及确切原始顺序。原始CSI/ATI大文件不在本包。
- WeightedRandomSampler按类别→房间平衡、有放回每epoch抽960次；unknown按记录identifier作组，不把它断言为同一物理房间。因此一轮不保证每个窗口都出现一次。batch32，AdamW lr0.0004、weight_decay0.01，梯度裁剪3；全量refit固定5轮，没有scheduler或early stopping。
- 损失=direct NLL +0.15专家CE +0.5条件序数BCE +0.1双头JS +0.1六项信号相干代理MSE；原run没有label smoothing。

不能只回答“没有增广”：原训练forward内实际包含：
1. 幅度/相位高斯噪声std0.02；5%频率格mask，沿时间共享，128格幅度用32格mask每格重复4次。
2. 静态分支前两个残差块后的MixStyle：批内置换，Beta(0.2,0.2)，训练时启用。
3. 四路线独立25%随机丢弃；全丢时恢复全保留；另有共享读出Dropout0.4与时间FFN Dropout0.1。
原run没有SAM、MixUp、CutMix、训练时Rx置换/时间反转、U伪标签或外部数据微调。后来的其他实验不能混算入此run。

original_source/下refit_structured.py、train_structured.py、structured_model.py、export_structured.py、validate_model.py来自原提交source_snapshot，逐文件校验归档hash；其余依赖来自同时期历史归档，不能保证全为运行当刻快照。详见provenance.json。

注意当前项目src/train.py是后期整理版，不应充当原run逐字节快照。原脚本依赖artifacts/tf_cache、phase_cache、baseline_a1_seed42的原特征银行；包内没有这些大缓存。直接从raw重算可能有浮点/批量/环境差异，不能承诺只凭seed42重训逐字节一致。final.pt不含优化器或RNG状态，不能精确续接历史优化轨迹。

当前0.54版本的区别：
- 原Gprime054：原底座权重按队友0.44文件的FP16存储精度匹配，使用队友给定五维偏置（估计过程未完整提供），6种Rx排列×正反时间，既有保护/融合规则及末尾1人+0.35。
- 选择性054：还复用seed42、paired方法第120步的相位/时间后缀，与初始权重按25%/75%插值并按原存储精度舍入；原Gprime最终预测2+保持，0/1区域使用适配专家。扩展checkpoint另放extensions/paired_seed42_step120.pt，仅含后缀state/initial，并非完整final.pt。
- 最新偏置054：在选择性054最终输出追加[-0.05,0,+0.15,-0.15,-0.15]，用户本次反馈仍0.54，非新的神经网络训练。原0.37 final.pt单独不包括这些后处理和额外后缀。

扩展paired训练仍只用同一960个有标签窗，无U/外部数据。seed42，源采样generator42、同房间配对generator20042；模型eval以冻结Dropout/MixStyle，训练最后phase.project、temporal.blocks.2、temporal.project。240步中保存40/120/240；实际选择120。AdamW编码器lr5e-5、weight_decay0.01、clip1，源batch16并额外8对同房间0/1；0.02标签平滑、0.1教师KL、0.02表示漂移、0.3配对binaryCE、0.1softplus排序。输入4视角，不等于原run训练时有TTA增广。历史原生特征缓存与实际ONNX预处理存在已发现的数值差异，不保证此旧扩展训练完全对齐最新ONNX口径。

本包提供证据和权重，不声称已完整复现队友未提供的偏置估计历史。平台成绩是用户报告，不是本次重新调用老师测试机所得。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    PACKAGE.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(PACKAGE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
        z.write(CP,'final.pt');z.write(PAIRED,'extensions/paired_seed42_step120.pt')
    with zipfile.ZipFile(PACKAGE) as z:assert z.testzip() is None;assert hashlib.sha256(z.read('final.pt')).hexdigest()==sha(CP);assert hashlib.sha256(z.read('extensions/paired_seed42_step120.pt')).hexdigest()==sha(PAIRED)
    save(OUT/'package_location.json',dict(path=str(PACKAGE),bytes=PACKAGE.stat().st_size,sha256=sha(PACKAGE),original_checkpoint_hash=sha(CP)))
    notice='最新平台反馈：重复记录校准偏置版0.54，未高于已有054；已记录为并列最高，暂停新实验，先交接原run种子/数据/增广/权重证据。以下旧待测状态为历史。'
    for filename in ('README.md','docs/SUMMARY.md','models/README.md','docs/ROBUST_BIAS054_20261025.md'):
        p=ROOT/filename;text=p.read_text(encoding='utf-8');heading,rest=text.split('\n',1)
        if notice not in text:p.write_text(heading+'\n\n'+notice+'\n'+rest,encoding='utf-8')
    print('HANDOFF_READY',PACKAGE,PACKAGE.stat().st_size,sha(PACKAGE),flush=True)
if __name__=='__main__':main()
