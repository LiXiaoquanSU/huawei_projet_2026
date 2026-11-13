"""Report schedule correction, learnability control and one structural ablation."""
import shutil
from train_strength import *
PRIOR=Path('D:/huawei2026-experiments/shared_hierarchy_20261025')

def summarize(folder):
    d=torch.load(folder/'predictions.pt',weights_only=True);z=d['logits'];y=d['labels'];p=torch.where(z[:,0].exp()>=.5,0,z[:,1:].argmax(-1)+1)
    r=read(folder/'result.json')
    return dict(joint=int((z.argmax(-1)==y).sum()),hard=int((p==y).sum()),epochs=r['selected_epoch'],history=r['history'],sha256=r['checkpoint_sha256'])

def main():
    selection=read(WORK/'selection.json');r=read(WORK/'results.json');meta=read(WORK/'published.json');check=meta['resource_check']
    controls={name:summarize(folder) for name,folder in [('原压缩计划3轮',PRIOR/'actual_plain_s2026'),('修正20轮计划、训练3轮',WORK/'actual_control_s2026'),('原结构相同轮数对照',WORK/'actual_control_matched_s2026'),('原结构训练20轮',WORK/'actual_control_long_s2026'),('加入幅度读出、源侧选轮次',WORK/'actual_strength_s2026')]}
    assert abs(controls['原压缩计划3轮']['history'][0]['loss']-controls['修正20轮计划、训练3轮']['history'][0]['loss'])<1e-5
    save(WORK/'control_comparison.json',controls)
    lines=['# 2026-10-26：学习率计划修正与绝对幅度读出','',
      '当前已评分最佳仍0.54。本轮是单变量结构对照和训练诊断；没有新的平台分数。','',
      '## 修正的训练细节','',
      '上轮代理在20轮余弦学习率计划中选出早期权重，但全数据只训练3轮时也把余弦周期压缩到3轮。两者学习率轨迹不同。修正后，训练到选中轮次即停，周期仍固定20轮。旧文件保留用于对照。',
      '同种子对照第一轮损失完全一致；后续只改变学习率轨迹。修正并未使3轮训练准确率提高，说明该问题需要修复，却不足以解释整个泛化失败。','',
      '## 本轮结构变量','',
      '原共享天线分层网络增加389参数的绝对幅度读出：从三天线的平均幅度、幅度标准差两组统计，各计算跨天线均值、标准差和最大值，得到6维输入；6→32→5网络为有人logit与四个人数logit提供最大±0.5的修正。输出层零初始化，保留原底座初始化和随机增强序列。',
      '该分支可能恢复有用的信号强度，也可能增加对房间的依赖；不是已证实的物理人数特征。继续检查6种天线重排不变性。',
      '四个留房间使用上轮源侧选择的seed2026，训练配方保持相同；每折仅用源侧独立记录NLL选轮次。固定0.5先判有人再选人数是本轮主要判决方式，未扫描阈值。全部48条记录的训练轮数为四折选中轮次中位数向下取整，余弦周期固定20。',
      '单种子和反复使用过的开发房间意味着本轮结论仍是初步证据；不称新盲测。','',
      '## 固定分层判决：相同880窗','',
      '|留出房间|上轮新结构|本轮信号强度读出|历史054代理|源侧选轮次|','|---|---:|---:|---:|---:|']
    for room,m in r['rooms'].items():lines.append(f'|{room}|{m["previous_correct"]}/{m["total"]}|{m["correct"]}/{m["total"]}|{m["baseline_correct"]}/{m["total"]}|{m["source_epoch"]}|')
    lines += ['',f'总计：上轮269/880，本轮{r["correct"]}/880（{r["correct"]/880:.2%}），历史054代理278/880（31.59%）。相对054的固定房间内记录bootstrap区间：{r["ci95"][0]:+.2%}至{r["ci95"][1]:+.2%}。不含多轮选型或新房间总体不确定性。',
      '类别诊断：本轮真实0/1/2/3/4人分别正确0/199/80/22/0窗，类别总数200/300/120/120/140。旧054代理对应0/225/51/2/0窗。两者在这些开发折中都未识别出0与4人，因此不能把这批代理上的总分提高直接当成平台能力提高。它提示验证数据覆盖与迁移代表性也需要重新审视。','',
      '## 全数据学习能力对照（全部是训练来源，不是独立测试）','',
      '|配方|训练轮数|联合概率正确/960|固定分层正确/960|','|---|---:|---:|---:|']
    for name,m in controls.items():lines.append(f'|{name}|{m["epochs"]}|{m["joint"]}|{m["hard"]}|')
    lines += ['',
      '同为6轮时原结构联合分类505/960、加入幅度读出637/960，支持新增绝对强度信息有用；但这仍是来源对照，无法保证新房间提升。20轮原结构达到842/960（87.71%），说明它具有拟合来源数据的能力。',
      '20轮全数据对照只检查模型是否能拟合来源数据，不能根据这个分数选择提交权重，也不能拿它与网站0.54直接比较。上轮四折源侧验证已有后期过拟合现象；多训练是否改善隐藏房间并未由该来源对照证明。','',
      '## 文件与结论','',
      f'- 新实验ONNX：{meta["path"]}；{meta["bytes"]/1e6:.2f}MB；SHA256 `{meta["sha256"]}`。',
      f'- 真实CSI覆盖0–4人和零输入数值核验通过；本机CPU4线程最慢平均{max(x["cpu_mean_s"] for x in check["cases"]):.3f}秒，资源检查通过。不是老师测试机速度保证。',
      '- 输出是固定阈值的分层判决分数，不是校准后的概率。没有外部数据、U伪标签或隐藏测试标签。',
      '- 源码：experiments/strength_ablation_20261026/；训练清单、8份权重与控制对照：D:/huawei2026-experiments/strength_ablation_20261026/。',
      '- 上轮文件未覆盖，当前最佳0.54未替换，未推送Git或上传平台。',
      ('本地通过推荐门槛，仍需平台验证，不能保证超过0.54。' if meta['recommended_over054'] else '本地证据未通过替换054的门槛，新文件只作为研究成果保留，不推荐消耗提交机会。')]
    report=ROOT/'docs/STRENGTH_ABLATION_20261026.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copyfile(report,ROOT/'2026-10-25_037配方与种子复现/信号强度与训练计划对照_20261026.md')
    note=f'最新研究（2026-10-26）：信号强度读出本地{r["correct"]}/880，上轮新结构269/880，054代理278/880；平台未测，最佳仍0.54。见[报告](docs/STRENGTH_ABLATION_20261026.md)。以下旧状态为历史。\n\n'
    for path in [ROOT/'README.md',ROOT/'models/README.md']:
        text=note if path.parent==ROOT else note.replace('(docs/','(../docs/');path.write_text(text+path.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(report,flush=True)

if __name__=='__main__':main()
