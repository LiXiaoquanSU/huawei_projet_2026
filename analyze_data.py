"""Audit official CSI/ATI and create deterministic, room-isolated manifests."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import blosc2
import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(rows):
    return {str(i): sum(str(r['headcount']) == str(i) for r in rows) for i in range(5)}


def audit(root, output):
    records, windows = [], []
    digests = defaultdict(list)
    for split in ['labeled', 'unlabeled']:
        with (root / split / 'samples.csv').open(newline='', encoding='utf-8-sig') as stream:
            index = list(csv.DictReader(stream))
        assert len({r['identifier'] for r in index}) == len(index), 'Duplicate identifiers'
        for n, row in enumerate(index):
            ident = row['identifier']
            csi_path = root / split / f'CSI_{ident}.b2'
            ati_path = root / split / f'ATI_{ident}.b2'
            csi = blosc2.load_array(str(csi_path))
            ati = blosc2.load_array(str(ati_path))
            if csi.shape != (12000, 1, 3, 1024) or csi.dtype != np.complex64:
                raise ValueError(f'Invalid CSI shape/dtype: {ident}')
            if ati.shape != (12000,) or ati.dtype != np.float32:
                raise ValueError(f'Invalid ATI shape/dtype: {ident}')
            if split == 'labeled' and row['headcount'] not in set('01234'):
                raise ValueError(f'Invalid label: {ident}')
            if split == 'unlabeled' and row['headcount'] != 'unlabeled':
                raise ValueError(f'Unexpected unlabeled label: {ident}')
            digest = hashlib.sha256(csi.tobytes()).hexdigest()
            digests[digest].append(f'{split}/{ident}')
            finite_csi = bool(np.isfinite(csi).all())
            finite_ati = bool(np.isfinite(ati).all())
            if not finite_csi or not finite_ati:
                raise ValueError(f'Non-finite data: {ident}')
            amp = np.abs(csi)
            intervals = ati[:-1]
            valid = intervals[intervals > 0]
            q = np.percentile(valid, [0, 1, 50, 95, 99, 100]) if len(valid) else np.zeros(6)
            mean = float(amp.mean(dtype=np.float64))
            # Dynamic statistics use every frame; do not interpret them as headcount.
            delta = np.diff(amp, axis=0)
            diff_rms = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            silent = np.all(csi == 0, axis=(0, 1))  # (Rx, subcarrier)
            rec = dict(split=split, **row, csi_path=str(csi_path), ati_path=str(ati_path),
                       n_frames=len(csi), csi_dtype=str(csi.dtype), ati_dtype=str(ati.dtype),
                       finite_csi=finite_csi, finite_ati=finite_ati,
                       duration_s=float(intervals.sum(dtype=np.float64) / 1e6),
                       effective_hz=float((len(csi)-1)*1e6/intervals.sum(dtype=np.float64)),
                       ati_min_us=float(q[0]), ati_p01_us=float(q[1]), ati_median_us=float(q[2]),
                       ati_p95_us=float(q[3]), ati_p99_us=float(q[4]), ati_max_us=float(q[5]),
                       ati_nonpositive_internal=int((intervals <= 0).sum()),
                       ati_last=float(ati[-1]), ati_gt_20ms_fraction=float((intervals > 20000).mean()),
                       amplitude_mean=mean, amplitude_std=float(amp.std(dtype=np.float64)),
                       amplitude_max=float(amp.max()), csi_zero_fraction=float((csi == 0).mean()),
                       silent_rx_subcarrier_bins=int(silent.sum()),
                       amplitude_diff_rms=diff_rms, relative_amplitude_diff=diff_rms/max(mean, 1e-12),
                       csi_sha256=digest)
            for rx in range(3):
                rec[f'rx{rx}_amplitude_mean'] = float(amp[:, 0, rx].mean(dtype=np.float64))
                rec[f'rx{rx}_silent_subcarriers'] = int(silent[rx].sum())
            records.append(rec)
            for start in range(0, 12000, 600):
                piece = amp[start:start+600]
                dt = ati[start:start+599]  # window boundary interval is excluded
                windows.append(dict(split=split, identifier=ident, layout=row['layout'],
                                    room=row['room'], headcount=row['headcount'], start=start,
                                    stop=start+600, duration_s=float(dt.sum(dtype=np.float64)/1e6),
                                    amplitude_mean=float(piece.mean(dtype=np.float64)),
                                    amplitude_std=float(piece.std(dtype=np.float64)),
                                    ati_max_us=float(dt.max())))
            print(f'{split}: {n+1}/{len(index)} {ident}', flush=True)
            del csi, ati, amp, delta
    duplicate_groups = [group for group in digests.values() if len(group) > 1]
    write_csv(output / 'trace_statistics.csv', records)
    write_csv(output / 'window_statistics.csv', windows)
    return records, windows, duplicate_groups


def make_splits(records, output, duplicates):
    labeled = [r for r in records if r['split'] == 'labeled']
    known = [r for r in labeled if r['room'] != 'unknown' and r['layout'] != 'unknown']
    excluded = [r for r in labeled if r not in known]
    fields = ['identifier', 'layout', 'room', 'acquisition', 'headcount', 'csi_path', 'ati_path']
    rooms = sorted({(r['layout'], r['room']) for r in known})
    summaries = []
    for layout, room in rooms:
        name = f'holdout_{room}'
        folder = output / 'splits' / name
        folder.mkdir(parents=True, exist_ok=True)
        val = [r for r in known if (r['layout'], r['room']) == (layout, room)]
        train = [r for r in known if (r['layout'], r['room']) != (layout, room)]
        assert not {r['identifier'] for r in train} & {r['identifier'] for r in val}
        assert not {(r['layout'], r['room']) for r in train} & {(r['layout'], r['room']) for r in val}
        assert not {r['csi_sha256'] for r in train} & {r['csi_sha256'] for r in val}, 'Duplicate CSI across fold'
        if any(v == 0 for v in counts(train).values()):
            raise ValueError(f'Training fold missing class: {name}')
        for part, rows in [('train', train), ('val', val), ('excluded_unknown', excluded)]:
            write_csv(folder / f'{part}_traces.csv', [{k: r[k] for k in fields} for r in rows])
            win = [dict(identifier=r['identifier'], layout=r['layout'], room=r['room'],
                        headcount=r['headcount'], csi_path=r['csi_path'], ati_path=r['ati_path'],
                        start=s, stop=s+600) for r in rows for s in range(0, 12000, 600)]
            write_csv(folder / f'{part}_windows.csv', win)
        summaries.append(dict(fold=name, role='primary' if room == 'a1' else 'secondary',
                              train_traces=len(train), val_traces=len(val),
                              train_windows=20*len(train), val_windows=20*len(val),
                              train_counts=counts(train), val_counts=counts(val),
                              validation_room=room, validation_layout=layout))
    settings = dict(data_root=str(Path(records[0]['csi_path']).parents[1]),
                    primary_fold='holdout_a1', window_frames=600, stride_frames=600,
                    interval_rule='slice start:stop; copy; set final element to zero',
                    unknown_policy='Exclude all unknown labeled traces from strict development folds',
                    unlabeled_policy='Not used by baseline; unknown room overlap cannot be ruled out',
                    preprocessing_policy='Fit learned statistics only on training traces of each fold',
                    duplicate_csi_groups=duplicates, folds=summaries)
    (output / 'split_config.json').write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding='utf-8')
    return summaries


def report(records, windows, folds, duplicates, output):
    labeled = [r for r in records if r['split'] == 'labeled']
    lines = ['# 数据分析与验证划分', '',
             '本报告扫描全部官方记录的完整 CSI/ATI；只读数据，不修改原始文件。未使用外部数据。', '',
             '## 人数与房间分布', '', '| 房间 | 布局 | 0人 | 1人 | 2人 | 3人 | 4人 | 记录数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for room in sorted({r['room'] for r in labeled}):
        group = [r for r in labeled if r['room'] == room]
        c = counts(group)
        lines.append(f'| {room} | {group[0]["layout"]} | ' + ' | '.join(str(c[str(i)]) for i in range(5)) + f' | {len(group)} |')
    lines += ['', 'unknown 是缺失元数据，不代表同一物理房间。', '', '## 信号质量与采样', '',
              '| 指标 | 有标签 | 无标签 |', '|---|---:|---:|']
    for title, func in [
        ('记录数', lambda rs: str(len(rs))),
        ('时长范围（秒/记录）', lambda rs: f'{min(r["duration_s"] for r in rs):.3f}–{max(r["duration_s"] for r in rs):.3f}'),
        ('有效采样率范围（Hz）', lambda rs: f'{min(r["effective_hz"] for r in rs):.2f}–{max(r["effective_hz"] for r in rs):.2f}'),
        ('记录内 ATI 中位数范围（微秒）', lambda rs: f'{min(r["ati_median_us"] for r in rs):.1f}–{max(r["ati_median_us"] for r in rs):.1f}'),
        ('最大帧间隔（毫秒）', lambda rs: f'{max(r["ati_max_us"] for r in rs)/1000:.2f}'),
        ('内部非正帧间隔总数', lambda rs: str(sum(r['ati_nonpositive_internal'] for r in rs))),
        ('末尾 ATI 非零记录数', lambda rs: str(sum(r['ati_last'] != 0 for r in rs))),
        ('平均幅度范围（原始设备单位）', lambda rs: f'{min(r["amplitude_mean"] for r in rs):.3g}–{max(r["amplitude_mean"] for r in rs):.3g}'),
        ('持续为零的 Rx×子载波数量范围', lambda rs: f'{min(r["silent_rx_subcarrier_bins"] for r in rs)}–{max(r["silent_rx_subcarrier_bins"] for r in rs)}'),
    ]:
        groups = [[r for r in records if r['split'] == s] for s in ['labeled', 'unlabeled']]
        lines.append(f'| {title} | {func(groups[0])} | {func(groups[1])} |')
    lines += ['', '- 全部记录形状和类型通过检查：CSI `(12000,1,3,1024)` complex64；ATI `(12000,)` float32。',
              '- 完整数组 NaN/Inf 检查通过。持续零值可能为无效频点，不能未经确认就认定损坏。',
              f'- 完整 CSI 数组 SHA-256 重复组数：{len(duplicates)}（跨训练/验证重复会触发失败）。',
              '- ATI 统计排除末尾补零；有效采样率为 11999 / 实际时长。600帧窗口不保证恰好3秒。',
              '', '### 按房间/无标签域的信号统计', '',
              '| 数据 | 房间标识 | 记录数 | 平均幅度中位数 | 相邻帧幅度差RMS/平均幅度中位数 | ATI中位数（微秒）中位数 |',
              '|---|---|---:|---:|---:|---:|']
    for split in ['labeled', 'unlabeled']:
        for room in sorted({r['room'] for r in records if r['split'] == split}):
            rs = [r for r in records if r['split'] == split and r['room'] == room]
            med = lambda k: float(np.median([r[k] for r in rs]))
            lines.append(f'| {split} | {room} | {len(rs)} | {med("amplitude_mean"):.4g} | {med("relative_amplitude_diff"):.4g} | {med("ati_median_us"):.1f} |')
    lines += ['', '这些是描述性统计，不能据此推断哪个无标签标识为室外，也不能将动态强度直接解释为人数。',
              '', '## 验证划分', '',
              '| 留出房间 | 训练记录 | 验证记录 | 训练窗口 | 验证窗口 | 验证人数类别 |',
              '|---|---:|---:|---:|---:|---|']
    for fold in folds:
        classes = ','.join(k for k, v in fold['val_counts'].items() if v)
        lines.append(f'| {fold["validation_room"]} | {fold["train_traces"]} | {fold["val_traces"]} | {fold["train_windows"]} | {fold["val_windows"]} | {classes} |')
    lines += ['', '**主验证集：a1。** 它是唯一同时包含0–4人的已知房间，12条记录、240个不重叠窗口；训练使用其他4个已知房间的33条记录、660个窗口。',
              '', '**辅助验证：c1/g1/b1/e1 逐房间留出。** c1覆盖1–4人，g1覆盖0–1人；b1只有1条0人记录，e1只有3条0–1人记录，这些小折仅作诊断，不单独代表五分类能力。',
              '', '**局限：** 每个已知布局在有标签数据中仅有一个房间，因此这些折同时留出了布局；比官方“训练布局族内新房间”更苛刻。当前数据无法构造已知同布局、不同房间的有监督验证。验证窗口相互相关，不能视为240次独立采集。',
              '', '## 后续训练约定', '',
              '1. 所有严格验证折排除3条房间未知的有标签记录，防止未知房间与验证房间重叠。未来全量最终训练可再考虑使用。',
              '2. 本阶段不把无标签记录加入训练；未来预训练单独设计和对照，尤其不能假设所有无标签记录都为室内。',
              '3. 先按记录/房间划分，再切窗口。窗口固定600帧、步长600；训练时可仅在训练记录内随机裁剪，验证窗口固定。',
              '4. 从 start:stop 切出 ATI 后复制并把最后一个值设为0，避免引入窗口外时间间隔。',
              '5. 数据集级归一化、PCA、特征选择和统计估计只在每折训练数据拟合；不要从全数据拟合。',
              '6. 主指标为固定窗口整体Accuracy，同时报告各房间Accuracy、各类别召回、混淆矩阵和房间宏平均。缺失类别记为不可评估，不能记为0。',
              '7. 不对验证集做过采样或随机增强；类别均衡仅用于训练。不要只优化a1，需检查多个留出房间的稳定性。',
              '', '## 输出文件', '',
              '- `trace_statistics.csv`：每条记录的完整信号质量和时间统计。',
              '- `window_statistics.csv`：所有固定600帧窗口的统计。',
              '- `split_config.json`：固定划分配置及人数分布。',
              '- `splits/holdout_*/{train,val}_traces.csv` 和 `*_windows.csv`：训练可直接读取的绝对路径与帧范围。',
              '- `excluded_unknown_*`：明确排除的未知房间记录/窗口，不参与严格验证。', '']
    lines += ['## 采样间断专项检查', '',
              '100毫秒仅为人工复查阈值，不是剔除标准。当前所有原始记录均保留；验证集也不因间隔较大而删样本。', '']
    flags = []
    for split in ['labeled', 'unlabeled']:
        ws = [w for w in windows if w['split'] == split]
        rs = [r for r in records if r['split'] == split]
        lo, hi = min(w['duration_s'] for w in ws), max(w['duration_s'] for w in ws)
        lines.append(f'- {split}：600帧窗口实际跨度 {lo:.4f}–{hi:.4f} 秒；含超过100毫秒间隔的窗口 {sum(w["ati_max_us"] > 100000 for w in ws)} 个。')
        for r in rs:
            if r['ati_max_us'] > 100000:
                flags.append({k: r[k] for k in ['split', 'identifier', 'room', 'headcount', 'ati_max_us', 'duration_s']})
    lines += ['', '所有记录每根接收天线都有11个持续为零的子载波（合计33个Rx×频点）；尚未据此确认所有记录的零值频点位置完全一致，不先硬编码删除位置。',
              '', '长间隔记录列于 `timing_review.csv`。模型应使用ATI或对间断做稳健处理，不能把帧序号直接等同均匀物理时间。', '']
    write_csv(output / 'timing_review.csv', flags)
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('D:/huawei2026-data'))
    parser.add_argument('--output', type=Path, default=Path('analysis'))
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, windows, duplicates = audit(root, output)
    folds = make_splits(records, output, duplicates)
    report(records, windows, folds, duplicates, output)
    print(f'Done: {len(records)} traces, {len(windows)} windows; {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
