"""Write factual architecture/training report; never infer platform score."""
import shutil
from train import *

def main():
    results=read(WORK/'results.json');sel=read(WORK/'selection.json');meta=read(WORK/'published.json');actual=read(WORK/'actual_metrics.json');check=meta['resource_check'];best=sel['development'];variant=sel['variant']
    lines=['# 共享天线、多尺度时间、分层人数网络','',
      '**已完成新结构训练与导出；平台尚未测试，不能声称达到0.54。** 当前最佳已评分模型仍为原Gprime0.54。','',
      '2026-10-25启动，2026-10-26完成；目录沿用启动日期。','', '## 模型构造','',
      '- 沿用已核对的复数CSI预处理，输入600帧、3根接收天线、1024子载波及时间间隔。预处理缓存来自真实054 ONNX；没有继承原037网络的学习权重。',
      '- 三根天线使用同一个编码器。输入为归一化幅度以及短时残差、中尺度变化、慢变化四种特征；5/17时间步平滑是在特征时间轴上，不宣称对应真实活动标签。',
      '- 三对天线分别使用cos/sin相位差，正反方向共6条共享编码输入。集合均值、标准差、注意力汇总保证天线重新编号不影响输出。无图消息传递、无12种接收排列推理。',
      '- 空间卷积后采用膨胀率1/2/4的时序卷积。背景均值/方差只通过±10%乘性调节影响节点表示；动态强度和时间质量统计参与融合。静态幅度路线仍含环境信息，因此不能声称完全去除了房间。',
      '- 主输出P(0)=1−P(有人)，P(k)=P(有人)×P(k|有人)，k=1..4。另有至少2/3/4人的辅助监督头，只参与训练，不在推理时重复投票。',
      '- 单模型514,970参数；两个独立种子均完成训练。开发中双种子融合失败，最终按源侧平均验证NLL选择seed2026单模型，取正反时间两次预测。旧偏置、旧保护规则与旧模型输出均未加入。','',
      '## 训练和验证','',
      '- 对照plain与crossroom两个版本；后者增加同人数、不同房间的对比约束。每个版本训练seed42和seed2026，每个种子分别留出a1/c1/g1/e1，共16次训练。',
      '- 每次最多20轮，24个窗口/批，每窗同时使用两种时间方向。AdamW学习率0.0006、weight decay0.02，余弦降到0.00006，梯度裁剪3。按人数×房间加权采样；unknown按记录区分。',
      '- 损失：人数NLL +0.2有序BCE +0.05正反时间一致性；crossroom另加0.03跨房间对比损失。特征噪声std0.01只作用于幅度/相位，不改时间。没有外部数据、U伪标签、旧模型蒸馏。',
      '- 整记录划分，fit/calibration/test记录互斥；留出房间完全不参加训练及轮次选择。训练侧独立calibration的类别平衡NLL选轮次，未使用目标房间标签挑轮次。',
      '- 两版本按两个种子平均留房间正确数选择。留房间数据已多轮用于开发，本轮选择仍属探索性，不是新盲测。最后使用所选版本源侧选中轮次的中位数，在全部48条记录960个非重叠窗口上训练两个完整模型。',
      '- 重要覆盖限制：留出a1时2/3/4人仅剩c1训练，留出c1时仅剩a1训练。因此这些折的多人类别没有跨房间同类正样本；约束不能凭空补齐缺失组合。unknown记录只用于人数训练/按记录采样，不被宣称是不同物理房间的对比正样本。',
      f'- 最终版本：{variant}；全数据训练轮数：{sel["epochs"]}；种子42、2026。','',
      '## 留房间结果（880窗）','',
      '|版本|seed42正确|seed2026正确|双种子融合正确|历史054代理正确|融合净增|',
      '|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f'|{r["variant"]}|{r["seeds"]["42"]["correct"]}|{r["seeds"]["2026"]["correct"]}|{r["correct"]}|{r["baseline_correct"]}|{r["net"]:+d}|')
    lines+=['','所选版本逐房间：','','|房间|新模型正确/总数|历史054代理正确|净增|','|---|---:|---:|---:|']
    for room,m in best['rooms'].items():lines.append(f'|{room}|{m["correct"]}/{m["total"]}|{m["baseline_correct"]}|{m["correct"]-m["baseline_correct"]:+d}|')
    cm=np.array([m['confusion_matrix'] for m in best['rooms'].values()]).sum(0)
    lines+=['','所选版本合计混淆矩阵：行是真实人数，列是预测人数。','','|真实人数|预测0|预测1|预测2|预测3|预测4|本类正确率|','|---|---:|---:|---:|---:|---:|---:|']
    for k in range(5):lines.append('|'+str(k)+'|'+'|'.join(str(v) for v in cm[k])+f'|{cm[k,k]/max(1,cm[k].sum()):.2%}|')
    lines += ['','分层诊断（有人判断以P(有人)>0.5；人数细分仅在真实有人窗口上看条件头）：','','|房间|有人判断正确/总数|真实有人时人数细分正确/总数|','|---|---:|---:|']
    for room,m in best['rooms'].items():lines.append(f'|{room}|{m["presence_correct"]}/{m["total"]}|{m["positive_count_correct"]}/{m["positive_total"]}|')
    lines += ['',f'最终选定单种子本地留房间准确率：{best["correct"]/best["total"]:.2%}；历史054代理：{best["baseline_correct"]/best["total"]:.2%}。固定房间内按记录bootstrap提升区间：{best["ci95"][0]:+.2%} 至 {best["ci95"][1]:+.2%}。该区间不含反复选型和新房间总体的不确定性。',
      '历史054代理使用对应留房间旧底座及其源侧校准，训练配方和预算与新模型不同；这是部署方法对照，不是仅改变一个结构变量的严格消融。平台0.54和本地代理准确率不是同一指标，不能直接比较数值。','',
      '## 全数据训练后的来源核对','',
      f'最终单种子在训练来源960窗上正确{actual["correct"]}，准确率{actual["correct"]/960:.2%}；原054在同批来源数据676/960。新模型相对它救回{actual["corrected"]}、改坏{actual["broken"]}。这些记录参加过训练，不是独立准确率。','',
      '## 导出和可复现文件','',
      f'- ONNX：{meta["path"]}；{meta["bytes"]/1e6:.2f}MB；SHA256 `{meta["sha256"]}`。',
      f'- 真实原始CSI覆盖0–4人和零输入核验；最大概率误差{max(x["probability_max_error"] for x in check["cases"]):.3g}。',
      f'- 本机CPU4线程，最慢用例平均{max(x["cpu_mean_s"] for x in check["cases"]):.3f}秒；峰值提交内存{check["peak_commit_bytes"]/1e9:.2f}GB。单文件嵌入权重，<100MB、<0.7秒、<6GB均经本机检查，不保证老师机器等速。',
      '- 输入frames float32[600,1,3,1024,2]、intervals float32[600]；输出logits float32[5]。',
      '- D:/huawei2026-experiments/shared_hierarchy_20261025/ 保存全部18份训练权重、逐轮日志、记录清单、960窗预测及导出核验；checkpoint与输入缓存SHA256已记录。',
      '- experiments/shared_hierarchy_20261025/ 保存完整新结构、训练、评估、导出和报告代码。未推送Git、未上传平台。','',
      ('本地整体指标改善，但是否达到/超过平台0.54必须实际提交验证。' if meta['recommended_over054'] else '本地证据不足以推荐替换当前0.54。ONNX作为可运行的新结构实验成果保留，不标记为已提升模型。')]
    lines+=['','单种子选择说明：预定双种子融合得到plain81/880、crossroom109/880，均出现严重下降。随后按四折训练侧最佳验证NLL的平均值选择交付种子：seed42为1.33399，seed2026为1.28436，故选seed2026。这是发现融合失败后的探索性调整，不是假称最初就预定单种子。']
    if (WORK/'hard_published.json').exists():
        hard=read(WORK/'hard_published.json');hr=hard['development'];hm=hard['source_metrics'];hc=hard['resource_check']
        lines+=['','## 补充：先判有人、再选人数的固定阈值实验','',
          '诊断发现c1的有人判断为240/240，但联合五分类仅1/240：条件人数分布过于分散，联合概率的最大值仍可能落在空房。用固定0.5阈值先作二分类，再在1–4中选人数，作为不再训练的后验对照。没有扫描阈值或借用隐藏测试标签。',
          f'该版本留房间{hr["correct"]}/880（{hr["correct"]/880:.2%}），各房间34/240、55/240、173/340、7/60；旧方法278/880。相对旧方法净少9窗，固定房间bootstrap区间{hr["ci95"][0]:+.2%}至{hr["ci95"][1]:+.2%}，不能称已追平或超过。',
          f'来源{hm["correct"]}/960（{hm["correct"]/960:.2%}），只比联合概率版多2窗；仍明显低于原054的676/960。输出为硬分层判决分数，不是校准后的五分类概率。',
          f'文件：{hard["path"]}；{hard["bytes"]/1e6:.2f}MB；SHA256 `{hard["sha256"]}`。真实CSI及零输入数值核验通过；本机最慢平均{max(x["cpu_mean_s"] for x in hc["cases"]):.3f}秒。',
          '这是本轮本地较好的实验输出，平台未测，仍不推荐替换当前0.54。两个ONNX和失败的融合结果均保留。当前只能判断这套结构与训练配方未达到目标，不能把原因单独归结为模型容量或某一个模块。']
    report=ROOT/'docs/SHARED_HIERARCHY_20261025.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(report,ROOT/'2026-10-25_037配方与种子复现/新结构分层人数训练报告.md')
    note=f'最新新结构实验已完成：联合概率版215/880；固定阈值分层版269/880；旧方法对照278/880。新模型平台未测，不推荐替换已评分0.54。详见[训练报告](docs/SHARED_HIERARCHY_20261025.md)。以下旧状态为历史。\n\n'
    for p in [ROOT/'README.md',ROOT/'models/README.md']:
        text=note if p.parent==ROOT else note.replace('(docs/','(../docs/')
        previous=p.read_text(encoding='utf-8-sig')
        if previous.startswith('最新新结构实验已完成：'):previous=previous.split('\n\n',1)[1]
        p.write_text(text+previous,encoding='utf-8')
    print(report,flush=True)

if __name__=='__main__':main()
