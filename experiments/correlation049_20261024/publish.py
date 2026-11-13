"""Register one verified covariance auxiliary candidate; preserve scored049."""
import csv,shutil
from run import *
from export import EXPORT,SOURCE,SOURCE_SHA,sha

def main():
    validation=read(EXPORT/'validation.json');resources=read(EXPORT/'resource_check.json');proof=read(EXPORT/'structure_proof.json')
    assert resources['status']=='pass' and sha(EXPORT/'model.onnx')==resources['sha256']==proof['candidate_sha256']
    assert sha(SOURCE)==SOURCE_SHA and validation['raw_windows']==960
    path=ROOT/'models/correlation_consensus_20261024/01_correlation_consensus.onnx';path.parent.mkdir(parents=True,exist_ok=True)
    if path.with_suffix('.json').exists():assert read(path.with_suffix('.json'))['platform_score'] is None,'Never overwrite a scored model'
    shutil.copy2(EXPORT/'model.onnx',path);assert sha(path)==resources['sha256']
    repeats=read(WORK/'no_clock_replicate_results.json');ensemble=[r for r in repeats if r['variant']=='ensemble' and r['room']!='actual']
    controls=read(WORK/'constant_controls.json');record=read(WORK/'ensemble_record_results.json');symmetry=read(WORK/'symmetry.json')
    predictions=read(EXPORT/'all_predictions.json');data=torch.load(WORK/'features.pt',weights_only=True);lookup={(r['identifier'],r['start']):i for i,r in enumerate(data['rows'])}
    with np.load(WORK/'no_clock_replicate_actual_ensemble.npz') as v:cached=v['logits']
    actual=np.array([r['logits'] for r in predictions],np.float32);reference=np.array([cached[lookup[r['identifier'],r['start']]] for r in predictions])
    def softmax(z):p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
    cachecheck=dict(cached_raw_argmax_disagreements=int((actual.argmax(-1)!=reference.argmax(-1)).sum()),probability_max_error=float(np.abs(softmax(actual)-softmax(reference)).max()))
    meta=dict(id='correlation_consensus_20261024',name='归一化相关性统计双种子辅助版',path=path.relative_to(ROOT).as_posix(),
      sha256=resources['sha256'],bytes=path.stat().st_size,platform_score=None,status='awaiting_platform',owner='self_statistical_auxiliary_on_scored049',
      source='new checked exploratory covariance branch; platform not evaluated',submission_id=None,report='docs/CORRELATION_CONSENSUS_20261024.md',
      base_sha256=SOURCE_SHA,base_platform_score=.49,new_classifier_parameters=19466,
      method='Frozen049 outputs preserved exactly on its predicted0/1. Otherwise0.5softmax049+0.25aux42+0.25aux2026. Aux114 normalized covariance/ACF descriptors; no clock/absolute summary.',
      experiment_root=str(WORK),local_validation=validation,resource_check=resources,independent_proxy_results=ensemble,
      independent_proxy_net=sum(r['metrics']['corrected']-r['metrics']['broken'] for r in ensemble),record_holdout=record,cache_raw_comparison=cachecheck,
      caveats='Exploratory. Most proxy gain concentrated g1, little independent high-count gain. Reused room development data. Combined source/record estimates inherit backbone label exposure. No guarantee to beat049.')
    save(path.with_suffix('.json'),meta)
    for name in ('structure_proof.json','validation.json','resource_check.json'):shutil.copy2(EXPORT/name,path.parent/name)
    registry=read(ROOT/'models/registry.json');assert registry['current_best']['platform_score']==.49
    registry['research_candidates']=[r for r in registry.get('research_candidates',[]) if r.get('id')!=meta['id']]+[meta]
    registry['next_submission']=meta;registry['research_status']='correlation_consensus_checked_awaiting_platform'
    registry['candidate_note']='Adapter platform048 below best049; new covariance/ACF auxiliary uses frozen049, keep01 protection, no clock summary. Candidate not scored.'
    registry['latest_research']=dict(report=meta['report'],status='new_statistical_branch_checked',best_scored=.49)
    save(ROOT/'models/registry.json',registry)
    p=ROOT/'models/platform_scores.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    rows=[r for r in rows if r['sha256']!=meta['sha256']];new={k:'' for k in fields}
    new.update(model=meta['name'],path=meta['path'],sha256=meta['sha256'],source=meta['source'],status='awaiting_platform',availability='retained',original_path=meta['path']);rows.append(new)
    with p.open('w',encoding='utf-8-sig',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save(ROOT/'checks/correlation_consensus_20261024.json',dict(metadata=meta,structure=proof,symmetry=symmetry,code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    classrows='\n'.join(f"|{k}|{old}|{validation['confusion_matrix'][k][k]}|{validation['confusion_matrix'][k][k]-old:+d}|" for k,old in enumerate([206,198,66,78,140]))
    roomrows='\n'.join(f"|{room}|"+'|'.join(f"{next(r for r in repeats if r['room']==room and r['variant']==name)['metrics']['corrected']-next(r for r in repeats if r['room']==room and r['variant']==name)['metrics']['broken']:+d}" for name in ('seed42','seed2026','ensemble'))+'|' for room in ('a1','c1','g1','e1'))
    text=f'''# 相关性统计辅助结构（2026-10-24）

## 状态

用户反馈上一份时间适配器`ec15ef...904`平台0.48，已录入侧记、registry及成绩表；原0.49仍最佳。源数据688→718没有迁移，停止增加该适配器强度。

新待提交文件：`{meta['path']}`，{meta['bytes']/1e6:.2f}MB，SHA256 `{meta['sha256']}`。尚无平台分数，不能宣称已超过0.49。

## 为什么换结构

固定权重的时间/频率平滑与局部100/120时间格双视图共12种设置，及直接/序数头4种非线性池化×3强度，收益有限。另比较3种冻结保护条件，没有选微小变化去强求新提交。全部预测与判定留在D盘multiscale049_20261024、ordinal_consensus_20261024。

新方向参考[CSI协方差人数研究](https://www.mdpi.com/1424-8220/18/11/3981)：多条载波是否一起变化、变化能量是否集中，有别于单点幅度值。论文采用特定同环境条件，不能照搬成绩，也不能把协方差秩等同于人数。

另核对了[跨环境增强研究](https://arxiv.org/html/2401.00964v1)，其增强收益有明显环境依赖；不是任何平滑、裁剪或幅度增强都会改善本题。

## 最终结构

原0.49网络、4个视角、内部路由、最终1人偏置+0.35全部保留，原9065节点与全部初始化权重字节保持。新增统计分支读取原视角预处理结果，不再额外运行CNN/Transformer编码器。

1. 动态幅度按3个Rx×32个频率格，相位按3个天线对×64个cos/sin格组织。
2. 每组时间中心化、按能量归一化，构建相关矩阵，提取归一化tr(C²)、tr(C³)、tr(C⁴)的压缩值。没有求特征值，也不把其中任何一项命名为真实人数。
3. 提取1/2/4/8/16/32时间格的自相关、时间能量集中度、半段能量差，以及时间差分的相关矩阵矩。
4. 对Rx/天线对作均值、最大、最小及联合统计，得到114维。最终新分支不读取绝对幅度summary或ATI summary；原主干仍正常使用真实ATI。新分支本身仍按包序列统计，不能说完全消除了采样率影响。
5. 两个114→64→32→5小网络（第一层后LayerNorm/SiLU/Dropout0.2，第二层SiLU），合计{proof['new_classifier_parameters']:,}个新可训练参数；种子42/2026的概率平均。

若原0.49最终预测0或1，**原5维分数逐字节直接返回**。其余输入输出概率为0.5×原0.49概率＋0.25×辅助42概率＋0.25×辅助2026概率，再取log。没有再加类别偏置，没有使用老师隐藏标签。

这是受限辅助，不是独立统计模型取代原网络。它不能修复原模型已经判为0/1的错误，尤其无法处理那94个源1→0中的任何一个。

## 训练与消融

只用比赛原有48条标注、960窗，没有外部/U/伪标签。统计特征不依赖预训练CNN。各折均值/标准差只从其fit记录计算，标准差最小0.05，输入标准化后截断±8。

AdamW lr0.001、weight_decay0.05、梯度截断1、按类别逆频数加权，全批训练400步并保留50/150/400快照；逐代理在其独立整记录calibration上选步数，全数据最终用这些选步数中位数对应的50步。辅助概率没有继承原0.44偏置，也没有单独拟合温度。

完成：4种统计头（有无绝对统计×softmax/条件序数）在4代理及全数据共20次；第二种子与单头整记录检查8次；去时钟5次；去时钟第二种子与整记录8次；最终双种子整记录6次，总计47次小读出训练。所有最终统计头为softmax，已有原主干CORN并非本轮新结构。

去时钟属于减少采样捷径的开发选择，不是证明原ATI一定有害。原119维版两种子代理净增14/11；去时钟114维版净增12/1，存在明显初始化敏感性，因此选两种子平均，而非宣称单一种子可靠。

## 独立房间代理

辅助统计头、其标准化及原主干教师均没有见外层房间标签；按既有fit/calibration/test整记录划分。主干校准保持各自原温度/偏置。代理房间此前反复用于开发，且留整房间同时留layout，与老师布局家族内跨房间有差异，不能称最终独立测试。

|外层房间|单头42净增|单头2026净增|最终平均净增|
|---|---:|---:|---:|
{roomrows}

平均版共净增10/880（约1.14个百分点），比上轮适配器的+2且g1−17更温和，但主要收益集中g1，并未证明高人数跨房间能力已有突破。

常量对照同用0.5混合与0/1保护：均匀分布净增0，fit标签比例净增−6，fit平均预测净增−9；实际输入统计净增10。源数据对应常量对照净增0/+10/+4，本候选净增{validation['corrected']-validation['broken']}。这支持收益需要输入相关输出，不能仅解释为加一份均匀先验；仍不能作因果人类动作归因。

## 原始输入验证

新ONNX实际运行全部960原始CSI/ATI窗：{validation['correct']}/960（{validation['accuracy']:.3%}），相对0.49纠正{validation['corrected']}、改坏{validation['broken']}。

|真实人数|原0.49正确|本候选正确|变化|
|---|---:|---:|---:|
{classrows}

这是训练来源诊断，不是独立准确率。实际原0.49对照分数与旧核查缓存最大差{validation['original049_logit_max_error']:.3g}；{validation['raw_protected_windows']}个原预测0/1窗口确认原分数精确保留。新统计头与独立PyTorch在实际原预处理特征上的计算逐窗比对，最大logit差{validation['independent_torch_auxiliary_logit_max_error']:.3g}。

缓存与原始导出预测不一致{cachecheck['cached_raw_argmax_disagreements']}窗，最大概率差{cachecheck['probability_max_error']:.3g}；正式汇报以上述原始输入结果为准。

31窗×6种Rx排列的新统计描述最大数值差约9.54e−7；特征时间反转核查通过。原始反转预处理缓存有约0.00175描述差，不能宣传整条候选严格Rx/时间对称。原0.49保护规则本身也不完全对称。

最终双头三折整记录排除：混合结果{record['mixed']['correct']}/960，纠正{record['mixed']['corrected']}、改坏{record['mixed']['broken']}；各折净变化{[r['mixed']['corrected']-r['mixed']['broken'] for r in record['folds']]}。第一折有退步，提示收益仍不稳定。统计头单独记录留出仅{record['head']['correct']}/960，不足以独立承担计数。混合基准主干历史见过源标签，所以混合留出指标仍不是端到端独立交叉验证。

## 资源与文件

单文件内嵌权重，CPU隔离检查通过：float32 frames[600,1,3,1024,2]与intervals[600]→logits[5]；4线程最慢用例平均{max(r['cpu_mean_s'] for r in resources['cases']):.3f}秒，峰值提交内存{resources['peak_commit_bytes']/1e9:.2f}GB，体积{resources['bytes']/1e6:.2f}MB，满足本地100MB/0.7秒/6GB检查阈值。

全部统计参数、独立代理、逐窗预测与对照在`{WORK.as_posix()}`；C盘仅保留这一份新候选与精简报告。未推送GitHub、未自动上传平台。仍需老师平台判断，当前最佳仍原0.49。
'''
    (ROOT/meta['report']).write_text(text,encoding='utf-8')
    intro=f"下一份待测：相关性统计辅助版`{meta['path']}`（{meta['bytes']/1e6:.2f}MB）。原始960窗{validation['correct']}正确，纠正{validation['corrected']}/改坏{validation['broken']}，属于训练来源诊断；独立房间代理净增10/880，主要在g1，平台未知，已评分最佳仍0.49。[本轮实验](docs/CORRELATION_CONSENSUS_20261024.md)。"
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8');lines=s.splitlines();lines=[line for line in lines if not line.startswith('本轮完成38次适配器训练，已导出并核查一个候选')];s='\n'.join(lines)+'\n';s=s.replace('# WiFi CSI 人数识别\n\n','# WiFi CSI 人数识别\n\n'+intro+'\n\n',1);p.write_text(s,encoding='utf-8')
    p=ROOT/'docs/SUMMARY.md';s=p.read_text(encoding='utf-8');s=s.replace('# 经验总结\n\n','# 经验总结\n\n'+intro.replace('docs/CORRELATION','CORRELATION')+'\n\n',1);p.write_text(s,encoding='utf-8')
    (ROOT/'models/README.md').write_text(f'''# 模型登记

最高已评分0.49：`final_bias_20261024/01_final_bias_one035.onnx`，字节不变。

下一份待测：`correlation_consensus_20261024/01_correlation_consensus.onnx`，{meta['bytes']/1e6:.2f}MB。归一化协方差/自相关统计双头辅助；平台分数未知。[实验记录](../docs/CORRELATION_CONSENSUS_20261024.md)。

时间适配器`temporal_adapter_20261024/01_temporal_adapter_consistent.onnx`已反馈0.48，停止推荐。其他获分里程碑保留，registry.json及platform_scores.csv为完整登记。空白分数表示尚未评测。

本轮未推送GitHub。新实验大文件在D盘，C盘只留一个新提交候选。
''',encoding='utf-8')
    print('PUBLISHED',path,validation,'best remains049',flush=True)
if __name__=='__main__':main()
