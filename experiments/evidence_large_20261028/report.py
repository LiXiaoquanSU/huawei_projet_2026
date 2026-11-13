"""Summarize all declared large-model development folds, no score-based cherry picking."""
import shutil
from train_chain import *

def main():
    totals={};rowsout=[]
    for family,folds in [('room',ROOMS),('record',FOLDS[4:])]:
        cm=np.zeros((5,5),dtype=int);last=cm.copy()
        for fold in folds:
            p=WORK/f'{fold}_full_s42';r=old.read(p/'result.json');cp=torch.load(p/'final.pt',weights_only=True)
            assert old.sha(p/'final.pt')==r['sha256']
            a,b,c=[set(cp[k]) for k in ('fit_records','calibration_records','test_records')]
            assert not(a&b or a&c or b&c)
            cm+=np.array(r['test']['confusion']);last+=np.array(r['last_test']['confusion'])
            rowsout.append((fold,r['epoch'],r['test']['correct'],r['test']['total'],r['last_test']['correct']))
        totals[family]=dict(correct=int(cm.trace()),total=int(cm.sum()),last_correct=int(last.trace()),confusion=cm.tolist(),last_confusion=last.tolist())
    actual=old.read(WORK/'actual_full_s42/result.json');sel=old.read(WORK/'selection.json')
    artifacts=[old.read(WORK/f) for f in ('published.json','published_last.json')]
    audit=dict(aggregates=totals,source=actual,selection=sel,platform_score=None,code_sha256={p.name:old.sha(p) for p in Path(__file__).parent.glob('*.py')})
    old.save(WORK/'research_results.json',audit)
    lines=['# 大参数天线关系识别链：2026-10-28','','已完成7组开发训练及1组全量训练，每组15轮，共120轮。参数10,874,074，约为小版186,554参数的58.3倍。平台未测。','',
    '## 实际扩展','','多尺度变化（原动态幅度、5点趋势残差、25点趋势残差）→保留4个频率区域的共享CNN→三接收天线节点及六条有向相位边→两轮节点/边消息交互→上下文及局部时间片的四层Transformer→全局上下文查询→五类联合读出。接收天线身份不使用位置编号编码，最终汇总对天线重排不变。','',
    '较小版新增了天线端点对应关系和更充分的上下文交互；仍使用旧确定性预处理，未加入原始有效性掩码，也没有恢复原先已平均掉的细节。局部概率不等于真实个人检测，时间尺度不等于有监督的活动类型。','',
    '## 训练与比较边界','','仅48条比赛标注记录、960个非重叠窗口，双时间视图。seed42，AdamW lr0.0006、weight_decay0.02、batch24、15轮余弦调度，NLL加0.05反转一致性损失；人数/房间/记录均衡采样。训练、校准和测试以整条记录隔离。源校准最低均衡NLL选epoch；七折选中epoch中位数取整决定全量选中权重，另保留固定15轮。没有旧学习权重、外部数据、U、输出偏置。','',
    '本轮同时改变了结构和容量，不能单独归因于参数增加。只有一个训练种子，开发数据已被反复使用，不能视为隐藏测试成绩。','',
    '## 开发结果','','|折|源选epoch|选中权重正确/总数|固定15轮正确/总数|','|---|---:|---:|---:|']
    for f,e,c,n,l in rowsout:lines.append(f'|{f}|{e}|{c}/{n}|{l}/{n}|')
    lines+=['','|协议|大版源选择|小版seed42源选择|大版固定15轮|小版seed42固定15轮|','|---|---:|---:|---:|---:|']
    for family,sc,sl in [('room',230,70),('record',422,491)]:
        a=totals[family];n=a['total'];lines.append(f"|{family}|{a['correct']}/{n} ({a['correct']/n:.2%})|{sc}/{n}|{a['last_correct']}/{n} ({a['last_correct']/n:.2%})|{sl}/{n}|")
    lines+=['','room是整房间隔离，record是整记录隔离（不保证同房间）。完整混淆矩阵、全部预测与两种检查点保存在D盘，禁止只用训练拟合率宣布泛化提高。','',
    '重要反例：g1源选择权重将340个窗口全部判成1人，因其中200个确为1人而得到200/340。a1主要预测1人或3人，c1主要预测4人；模型仍存在明显的整记录/环境相关类别偏向。留房间总数380/880高于小版230/880，不代表全面学会计数；整记录405/960反而低于小版422/960。应继续检查优化稳定性、类别召回及状态分支是否压过动态证据，而不是把容量增大直接当成机制进步。','',
    '## 全量训练及交付','',f"全量源选epoch{sel['epochs']}：{actual['test']['correct']}/960；固定15轮：{actual['last_test']['correct']}/960。这两项是训练拟合率。",'']
    for m in artifacts:
        check=m['resource_check'];lines.append(f"- `{m['path']}`：{m['bytes']:,}字节，SHA256 `{m['sha256']}`，CPU最慢样例平均{max(c['cpu_mean_s'] for c in check['cases']):.4f}秒，资源检查{check['status']}。")
    lines+=['','两份均为未评分研究模型。原平台最佳0.54保留。ONNX检查使用真实原始CSI及零输入，核对PyTorch/缓存一致性、嵌入权重、输入输出、CPU时间与内存；通过本地检查不保证平台运行速度或评分。','',
    '实验目录：`D:/huawei2026-experiments/evidence_large_20261028/`。代码：`experiments/evidence_large_20261028/`。']
    p=ROOT/'docs/EVIDENCE_LARGE_20261028.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(p,ROOT/'2026-10-25_037配方与种子复现/大参数关系链训练_20261028.md')
    print(json.dumps(dict(totals=totals,source_selected=actual['test'],source_last=actual['last_test'])),flush=True)

if __name__=='__main__':main()
