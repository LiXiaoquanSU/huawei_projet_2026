"""Copy verified candidate and update current team/project records."""
import csv,shutil
from run import *
PACKAGE=ROOT/'2026-10-25_037配方与种子复现'

def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)
def csvsave(p,fields,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    m=read(WORK/'published.json');s=m['selection'];check=m['resource_check'];assert check['status']=='pass' and s['robust_pass']
    src=ROOT/m['path'];assert sha(src)==m['sha256'];folder=PACKAGE/'新候选模型';folder.mkdir(exist_ok=True)
    dest=folder/src.name
    if dest.exists():assert sha(dest)==m['sha256']
    else:shutil.copy2(src,dest)
    save(dest.with_suffix('.json'),m);assert sha(dest)==m['sha256']
    max_time=max(c['cpu_mean_s'] for c in check['cases']);peak=check['peak_working_set_bytes']/1e6
    tail=f'''\n## 交付与提交文件\n
当前待测文件：`{m['path']}`；队友包内也有同名文件`新候选模型/{src.name}`。SHA256：`{m['sha256']}`。

大小{m['bytes']/1e6:.2f}MB，独立CPU四线程最慢案例平均{max_time:.4f}秒，峰值工作集{peak:.1f}MB，七个真实/零输入案例全部类别一致，概率最大误差小于0.000001。权重嵌入单个ONNX，无外部文件。

本轮模型没有重新训练基础权重；同一套编码器提取十二视角证据，在共享参数的人数读出中分别使用全路线、mask时间路线的输入。后者仍让时序参与presence和原始gate计算，只去掉具体人数读出的共享内容及独立专家中的直接时间路线。两者各用原054校准与保护规则，最终概率各50%。

四个留房间来源共880窗，基线278正确，新方案310正确：纠错43、破坏11、净增32（+3.64个百分点），四房间均改善。整记录bootstrap约+0.68～+6.93个百分点，但开发反复使用且未校正多次方案筛选。

来源训练诊断676/960→657/960（70.42%→68.44%），反而下降，已保留此负面结果。不能把开发提升推算为平台提升；新模型平台未测，不是已达到0.60的模型。

上轮seed310候选已登记平台0.53；三份已测0.54原文件及最佳记录全部保留。
'''
    p=ROOT/'docs/SUBNET_EXPECTATION_20261025.md';text=p.read_text(encoding='utf-8').split('\n## 交付与提交文件')[0]+tail
    p.write_text(text,encoding='utf-8');(PACKAGE/'后续实验_时间原点与子网络.md').write_text(text,encoding='utf-8')
    reportlink='docs/SUBNET_EXPECTATION_20261025.md'
    for rel in ['README.md','models/README.md']:
        f=ROOT/rel;text=f.read_text(encoding='utf-8');heading,rest=text.split('\n',1)
        notice=f'\n\n当前：上轮跨种子修正版平台0.53，已测最佳仍0.54；下一份待测为共享主干双人数读出`{src.name}`（{m["bytes"]/1e6:.2f}MB），四留房间+3.64个百分点，平台未知。报告见'+ ('../'+reportlink if rel.startswith('models/') else reportlink)+'。以下旧“最新/待测/暂停”文字为历史记录。\n'
        if notice.strip() not in text:f.write_text(heading+notice+rest,encoding='utf-8')
    f=ROOT/'models/platform_scores.csv';fields,rows=csvrows(f)
    r={k:'' for k in fields};r.update(model=m['name'],platform_score='',path=m['path'],sha256=m['sha256'],source=m['source'],status=m['status'],availability='retained',original_path=m['path'])
    rows=[x for x in rows if x['sha256']!=m['sha256']]+[r];csvsave(f,fields,rows)
    f=PACKAGE/'新增模型清单.csv';fields,rows=csvrows(f);rows=[r for r in rows if r['sha256']!=m['sha256']]
    rows.append({'文件':'新候选模型/'+src.name,'平台分':'未测试','状态':'本地筛选/资源检查通过的下一份待测','sha256':m['sha256']});csvsave(f,fields,rows)
    f=PACKAGE/'新候选模型/先读说明.md'
    f.write_text(f'当前下一份待测是`{src.name}`，平台尚未测试。上轮`02_exploratory_s310_adjacent_up_rx2.onnx`已测0.53，不是当前待测首选。已测最佳仍0.54。新方案本地四留房间+3.64个百分点，来源训练准确率反而下降，不能承诺平台0.60。详见上级后续实验_时间原点与子网络.md。\n',encoding='utf-8')
    f=PACKAGE/'README.md';text=f.read_text(encoding='utf-8').split('\n## 当前下一份模型')[0]
    text+=f'\n## 当前下一份模型\n\n上轮跨种子方案平台0.53，保留最佳0.54。下一份是`新候选模型/{src.name}`：共享主干双人数读出，{m["bytes"]/1e6:.2f}MB，已过CPU/数值检查，四留房间+3.64个百分点，平台尚未测。完整成功/失败见`后续实验_时间原点与子网络.md`。\n'
    f.write_text(text,encoding='utf-8')
    f=ROOT/'docs/TEAM_PROGRESS_FULL_20261025.md';text=f.read_text(encoding='utf-8').split('\n## 10. 跨种子053之后')[0]
    text+='\n## 10. 跨种子053之后：保留时序presence的双人数读出\n\n上轮平台0.53。新做时间原点12种方案，最好仅净增1/880，不导出；训练子网络期望/路线干预14种方案，选中no_time_all_0.5。完整保留静态/动态/相位/时间编码，同一主干的两种人数读出分别保留/丢弃时间路线的直接分类输入，presence和gate仍读时序，各50%概率融合。四留房间全部改善，278/880→310/880（+3.64pp），来源676/960→657/960下降。'+f'导出{m["bytes"]/1e6:.2f}MB，通过资源/数值检查，登记为待测，平台未知；已测最佳仍0.54。详情见docs/SUBNET_EXPECTATION_20261025.md。\n'
    f.write_text(text,encoding='utf-8');shutil.copy2(f,PACKAGE/'团队完整总结_含小全实验.md')
    # Keep frozen explicit recipe and26seed table unchanged.
    print(json.dumps(dict(candidate=m['path'],package_copy=str(dest),sha256=m['sha256'],cpu_max_mean=max_time,bytes=m['bytes'],platform_score=None),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
