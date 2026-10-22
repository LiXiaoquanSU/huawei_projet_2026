"""Finalize the data study and preserve evidence without changing model rankings."""
import csv,hashlib,json,shutil,zlib
from collections import Counter
from pathlib import Path
import numpy as np
from prepare import WORK,save
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    radio=json.loads((WORK/'radiopoints_audit.json').read_text());sharp=json.loads((WORK/'sharp_public/audit.json').read_text())
    for item in radio+sharp:
        p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
        raw=p.read_bytes()
        if 'git_blob' in item:assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==item['git_blob']
        if 'crc32' in item:assert zlib.crc32(raw)&0xffffffff==item['crc32']
    # Remove only this script run's redundant/interrupted matrix copies. The
    # downloaded MATs and the completed cleaning indices are preserved.
    clean=(WORK/'sharp_public/clean').resolve();removed=[]
    for item in sharp:
        stem=Path(item['path']).stem;p=(clean/f'{stem}.npz').resolve()
        assert p.parent==clean
        assert (clean/f'{stem}_index.npz').exists()
        if p.exists():removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
    old=json.loads((ROOT/'checks/restarts_cleanup_20261020.json').read_text(encoding='utf-8'))['model_hashes']
    for name,expected in old.items():assert sha(ROOT/name)==expected,name
    best={'models/best_037/model.onnx':'0fb02b7410899ad24be99b0b4d7a4077b3de106da3666dc054fc4e1d4e299aed','models/random_restart_20261020/02_original_seed42.onnx':'c4fa83755c11a057e970649004ad0fb4dd582dc98fb7acce569ec28714270e74'}
    for name,expected in best.items():assert sha(ROOT/name)==expected
    support=json.loads((WORK/'official_support.json').read_text())['record_counts']
    u=list(csv.DictReader(Path('D:/huawei2026-data/unlabeled/samples.csv').open(encoding='utf-8-sig')));uc=Counter(r['room'] for r in u)
    pilot=json.loads((WORK/'pilot_results.json').read_text());sp=json.loads((WORK/'sharp_public/pilot_results.json').read_text())
    rfeatures=np.load(WORK/'radiopoints_temporal_features.npz');sfeatures=np.load(WORK/'sharp_public/temporal_features.npz');ofeatures=np.load(WORK/'official_temporal_features.npz')
    for d in (rfeatures,sfeatures,ofeatures):assert d['x'].shape[1]==96 and np.isfinite(d['x']).all()
    snapshot=WORK/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snapshot/p.name)
    checks=dict(status='data adapted; full DataPort files require subscription/login',radiopoints_files=len(radio),radiopoints_bytes=sum(r['bytes'] for r in radio),radiopoints_windows=len(rfeatures['y']),sharp_public_files=len(sharp),sharp_public_bytes=sum(r['bytes'] for r in sharp),sharp_public_windows=len(sfeatures['y']),official_feature_windows=len(ofeatures['y']),official_records=48,official_U_records=len(u),official_U_rooms=dict(uc),radio_sha_and_git_blob_verified=True,sharp_sha_and_zip_crc_verified=True,old_model_hashes_verified=len(old),best_models=best,platform_best=.37,new_submission_model=False,deep_training_runs=0,diagnostic_rf_fits=len(pilot['results'])+len(sp['results']),removed_redundant_outputs=removed)
    save(ROOT/'checks/room_support_20261021.json',checks);save(WORK/'final_audit.json',checks)
    table='\n'.join('|'+room+'|'+'|'.join(map(str,counts))+'|' for room,counts in support.items())
    rtable='\n'.join(f"|{room}|"+'|'.join(str(next(v['windows'] for v in radio if v['room']==room and v['count']==y)) for y in range(5))+'|' for room in ('RoomA','RoomB','RoomC'))
    mainresults=pilot['results'];ftable='\n'.join(f"|{room}|"+'|'.join(f"{next(v['accuracy'] for v in mainresults if v['fold']==room and v['variant']==variant):.1%}" for variant in ('official_only','official_plus_radiopoints'))+'|' for room in ('a1','b1','c1','e1','g1'))
    external='、'.join(f"{v['fold']} {v['accuracy']:.1%}" for v in mainresults if v['variant']=='radiopoints_leave_room_out')
    stable='\n'.join(f"|{room}|"+'|'.join(f"{next(v['accuracy'] for v in sp['results'] if v['room']==room and v['task']==task):.1%}" for task in ('presence','activity'))+'|' for room in ('bedroom','living_room','laboratory'))
    report=f'''# 房间覆盖、外部数据与后续训练（2026-10-21）

结论：主要缺口是房间与人数高度耦合。增加模型容量、重复采样不能创造缺失场景；外部数据需要先证明能跨硬件迁移。本轮下载清洗了RadioPoints 15条0—4人记录，并取得作者另外公开的SHARP 13个MAT（0/1人及真实活动标签）。19次固定参数随机森林诊断已完成，0.37结构尚未重新深度训练，也没有新的提交ONNX。当前两份平台最佳仍为0.37。

## 官方数据重审

下表单位是独立原始记录，不是切出的窗口。

|房间|0人|1人|2人|3人|4人|
|---|---:|---:|---:|---:|---:|
{table}

共48条，5个可识别有标签房间；unknown的3条不能视作同一间房。2—4人只存在于a1、c1，c1没有空房。b1仅一条空房，e1/g1只有0/1。把g1的窗口大量重复，仍然没有g1的2/3/4人证据。当前基准已经按类别及该类别内房间平衡采样；再次声称“改成类别平衡”是新方法并不准确。

官方U共150条，按房间记录数：{dict(uc)}。6个已知新房间另加5条unknown。U没有人数真值。题目允许U与隐藏测试房间重合，S与隐藏测试房间不重合；不能从这些元数据推断测试各类人数比例、运动比例或具体内容。

开发时整房间留出，unknown不进入开发训练，以免无法验证房间隔离。最终方案确定后使用全部48条有标签记录，包括unknown三条；150条U可用于不需要人数标签的训练，但不把模型预测直接当真值。窗口切分后数据量增加，独立房间与采集次数没有增加。

## 取得并适配的数据

[RadioPoints原仓库](https://github.com/RadioPoints/Device-free_RF_Human_Sensing_Datasets)：固定commit/tree SHA保存在github_tree.json。仅下载RoomA/B/C各0—4人，共{sum(v['bytes'] for v in radio):,}字节；不将5—8人强行标成4人。保留原始二进制文件，SHA256及Git blob复核通过。

实际解码为2Tx×3Rx×30，20MHz，各条中位采样率约53—60Hz，少量包天线维度变化及文件截断尾部有单独计数。按作者推荐Intel csitool的格式解码、天线排列、RSSI/噪声缩放；格式参考源码保存在D盘，没有执行下载的代码。按实测timestamp_low微秒值处理时间，保留真实链路。3秒不重叠切窗，覆盖不足或超过100ms断包窗排除，得到{len(rfeatures['y'])}窗：

|房间|0人|1人|2人|3人|4人|
|---|---:|---:|---:|---:|---:|
{rtable}

每个房间每种人数仅一条原始采集；这是15条独立记录，不能宣称797条独立实验。没有明确坐姿/走路/跑步标签，不补造活动标注。仓库README注明非商业研究并要求引用两篇论文，另有GPL-3.0文件；两者原文都保留，不能仅凭代码许可证认定比赛使用许可。

[80MHz论文](https://arxiv.org/abs/2305.03170)与[正式数据页](https://ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels)：论文总量23.6GB，现网页文件显示22.02GB。网页明确要求订阅登录，访问证据在dataport_access.json，因此完整数据尚未下载。不能把论文或网页下载成功当成原始数据已取得。

七种环境并不都有0—4人计数：AR部分多环境但为0/1人；PC1—4全部同一会议室、不同硬件位置/遮挡，0和2—10人；1人需对应PI1—4。将来取得数据后只取PC的n00/n02/n03/n04，以及相应PI单人记录；AR的E标0，实际单人活动标1。四套同时采集的同后缀文件必须同组分折，不能随机拆开造成同步采集泄漏。原始矩阵与行排列需逐文件核查后再适配。

另找到作者[大学公开SHARP存档](https://researchdata.cab.unipd.it/624/)，它是相关较早数据，不冒充上述完整DataPort版本。选择S1a卧室、S6a客厅、S7a实验室的E/L/W/R（S7的L分两文件），13个MAT，{sum(v['bytes'] for v in sharp):,}字节；每个文件ZIP CRC和SHA256复核通过。原MAT实测容器为csi_buff二维复数256列，按作者公开预处理源码fftshift、去掉14个控制/边缘频点，保留242个有效频点；四天线交错行恢复为1Tx×4Rx。没有真实包时间戳，使用论文约6ms名义间隔并在每行清单显式标注。清洗索引不伪装为实测时间，不估计真实运动速度。

SHARP得到{len(sfeatures['y'])}个名义3秒窗，含空房、坐姿、走路、跑步。L是坐姿采集，不保证完全没有小动作；W不能改标成“缓慢移动”。只提供0/1人数，不能直接补2—4人。该公开集可能与DataPort的AR部分重合，将来不能把两者无去重合并为独立数据。公开数据页许可为ODC Attribution，参考代码的GPL另行保留。

适配层使用16种时间统计×6个真实链路/频点分位数，96维：相对幅度变化、五个时延相关、三个时延差分、五段幅度调制谱能量、谱熵。每链路最多选32个真实频点，低通重采样到50Hz。没有把30/242频点“补成1024”，没有复制天线，也没有把幅度调制谱叫作已经校正的物理多普勒。这些辅助特征可跨格式计算，但其物理统计分布仍可能不同。

## 已做的诊断与边界

随机森林固定192棵、深度10、叶最少3、种子42；整记录/整房间隔离，未根据验证调参。官方辅助特征955窗覆盖全部48条记录，5个断包或时长不足窗口仅在本轮诊断排除；原始数据及现有深度模型缓存不改。对照的外部样本总损失权重为官方的0.25，类别—房间—记录分层平衡。

RadioPoints留一房间五类准确率：{external}。外部数据自身有一定跨房间计数信息；只有三房间一轮采集，泛化证据仍有限。

|官方留出房间|仅官方时间描述|加入RadioPoints|
|---|---:|---:|
{ftable}

直接混入没有一致改善。b1只有空房，其90%不是五类计数90%；a1/c1的多人数仍很差。该RF不是0.37深度模型，两种成绩不可直接比较。此结果否定的是本次简单特征混合的可靠收益，不证明外部数据或所有迁移方案都无效。

SHARP整房间诊断：

|留出房间|0/1存在判断|空房/坐姿/走路/跑步四类|
|---|---:|---:|
{stable}

存在判断的高分只适用于这个外部0/1任务，不能当作老师五类成绩。坐姿/步行/跑步可提供辅助表示监督；步行与跑步仍有混淆，不能宣称准确恢复速度。论文SHARP也强调其动态识别范围；完全静止人体与空房在无已知背景参考的短CSI窗里可能难分，不能靠多普勒保证解决。

## 建议怎样训练

1. **先保留0.37结构，改训练目标。** 已有四路与序数/存在监督，不再次堆同义分支。先重现实验预算，保持相同更新数和初始化种子，试训练期的同人数跨房间约束。开发模型必须从随机或本折源数据预训练开始，不能加载见过留出房间标签的全量0.37权重。
2. **按“类别→有该类的房间→原始记录→时间窗”组织批次，同时保证全记录覆盖。** 现有加权替换采样并不保证每轮见过每条；先让每条源记录至少取一窗，再做平衡补采样。与相同总更新数的原采样器比较，避免把更多计算误当方法效果。不能填造缺失的房间×人数单元。
3. **同人数对齐，缺失人数不对齐。** 比如a1的3人与c1的3人做正对；a1的3人与g1空房不能为了“房间无关”强行拉近。0类可比较a/b/e/g，1类a/c/e/g，2—4类仅a/c。仅在训练折至少两个房间有该类时启用类条件原型/对比约束；unknown只参与人数监督，不参与房间配对。小权重约束与现有计数损失并行，由留房间验证决定是否保留。标签分布不一致时，盲目对齐整体域分布可能伤害判别，参见[ICML2019分析](https://proceedings.mlr.press/v97/zhao19a.html)。
4. **外部数据先独立编码，验证迁移再并入。** RadioPoints用于多房间0—4计数；SHARP用于活动/存在辅助任务；Brescia补真实不同人数但仍有单房间、日期及天线缺失限制。允许数据源自己的输入编码层，训练共享时间表征，官方完整幅度/相位/时间路线保留。先测官方源数据对照与RadioPoints迁移，再单独加SHARP辅助；只通过整房间验证才全量训练。初次试小外部权重如0.1，对照须匹配更新数，不能把这视为已证实最优比例。
5. **150条U先做不需要人数的学习。** 可测试同一窗口不同真实天线/时间遮挡的一致性或掩码预测；不得把不同窗口默认当成相同人数。先前重建预训练并未稳定优于基准，必须保留同预算纯监督对照。人数伪标签只可在多种子/增强一致、每类覆盖可审计时作为后续假设，禁止强行给每个U房间补齐0—4类。
6. **选择标准针对真正缺口。** a1整房间留出与c1整房间留出重点看2/3/4各类召回和五类混淆；g1/e1看0/1与误报高人数，b1仅看空房误报。报告各折实际人数支持，不能用b1高分掩盖多人失败。开发方案至少复验两个初始化种子；最终再用全部48条有标签数据训练，U与外部数据可作为明确标注的辅助来源。老师测试集未知，不能承诺0.5，也不能把反复看过的本地开发折叫盲测。

推荐下一组最小对照：A原结构原训练；B仅换覆盖优先采样；C在B上加官方同人数跨房间约束；D在C上加入RadioPoints独立编码迁移；通过后才增加SHARP活动辅助与U预训练。每步保留上一步作为对照，不一次叠满后无法判断作用。以上是后续方案，**本轮尚未运行这些深度网络训练**。

完整原始文件、清洗数据、逐窗清单、预测、混淆矩阵与源码快照均在D盘room_support_20261021。项目代码在experiments/room_support_20261021，轻量审计checks/room_support_20261021.json。旧{len(old)}份ONNX校验通过，两份平台0.37文件字节未改，平台记录与最佳排名未改。
'''
    (ROOT/'docs/ROOM_SUPPORT_20261021.md').write_text(report,encoding='utf-8')
    (WORK/'REPORT.md').write_text(report,encoding='utf-8');print(json.dumps(checks,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
