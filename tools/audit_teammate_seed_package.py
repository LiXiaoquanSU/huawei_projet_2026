"""Read-only validation of received seed package; retain evidence separately."""
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model import StructuredModel

PACKAGE = ROOT / '2026-10-25_037配方与种子复现'
OUT = ROOT / 'checks/teammate_seeds_20261025'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(PACKAGE / 'weights/weights_manifest.csv')
    stats = read_csv(PACKAGE / 'seeds_26_full_data.csv')
    scores = read_csv(PACKAGE / 'platform_scores.csv')
    original = torch.load(ROOT / 'models/best_037/final.pt', map_location='cpu', weights_only=True)
    model = StructuredModel('spectral', 'physical').eval()
    original_state = original['model_state']
    entries = []
    graph_signatures = []
    for row in manifest:
        p = PACKAGE / row['file']
        assert p.resolve().is_relative_to(PACKAGE.resolve())
        actual_hash = digest(p)
        assert actual_hash == row['sha256'], p
        assert p.stat().st_size == int(row['bytes']), p
        result = dict(seed=int(row['seed']), type=row['type'], file=row['file'],
                      bytes=p.stat().st_size, sha256=actual_hash, hash_matches=True)
        if row['type'] == 'checkpoint':
            checkpoint = torch.load(p, map_location='cpu', weights_only=True)
            state = checkpoint['model_state']
            model.load_state_dict(state, strict=True)
            assert all(torch.isfinite(t).all() for t in state.values()), p
            assert set(state) == set(original_state)
            same = [k for k, t in state.items() if torch.equal(t, original_state[k])]
            result.update(keys=list(checkpoint), config=checkpoint.get('config'),
                          checkpoint_seed=checkpoint.get('seed'), fold=checkpoint.get('fold'),
                          strict_load=True, finite=True, tensors=len(state),
                          tensors_identical_to_original=len(same),
                          identical_to_original=len(same) == len(state))
            if checkpoint.get('seed') is not None:
                assert int(checkpoint['seed']) == int(row['seed'])
            del checkpoint, state
        else:
            graph = onnx.load(p)
            onnx.checker.check_model(graph)
            signature = hashlib.sha256(b''.join(n.SerializeToString() for n in graph.graph.node)).hexdigest()
            graph_signatures.append(signature)
            result.update(onnx_valid=True, nodes=len(graph.graph.node),
                          initializers=len(graph.graph.initializer), graph_nodes_sha256=signature,
                          inputs={v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
                                  for v in graph.graph.input},
                          outputs={v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
                                   for v in graph.graph.output})
            del graph
        entries.append(result)
        print('VERIFIED', row['seed'], row['type'], flush=True)
        gc.collect()
    good = [int(r['seed']) for r in stats if float(r['b_TV']) <= .25]
    collapsed = [int(r['seed']) for r in stats if float(r['b_p0_%']) > 40]
    for r in stats:
        assert (r['good_TV_b<=0.25'] == 'True') == (int(r['seed']) in good)
        assert (r['collapsed_p0_b>0.40'] == 'True') == (int(r['seed']) in collapsed)
        for prefix in ['raw', 'b']:
            fractions = np.array([float(r[f'{prefix}_p{k}_%']) / 100 for k in range(5)])
            assert abs(.5 * np.abs(fractions - .2).sum() - float(r[f'{prefix}_TV'])) < .002
    assert len(manifest) == 52 and len(stats) == 26
    assert {int(r['seed']) for r in stats} == {0, 1, 2, 3, 4, 42, *range(300, 320)}
    summary = dict(package=str(PACKAGE), files_checked=len(entries),
                   bytes_checked=sum(r['bytes'] for r in entries),
                   all_hashes_match=True, checkpoints_strict_load=26, onnx_checks_passed=26,
                   node_graph_signatures=len(set(graph_signatures)),
                   original_checkpoint_sha256=digest(ROOT / 'models/best_037/final.pt'),
                   good_distribution_seeds=good, collapsed_distribution_seeds=collapsed,
                   overlapping_seeds=sorted(set(good) & set(collapsed)),
                   mean_room_p0_after_fixed_bias={room: float(np.mean([float(r[f'b_p0_{room}_%']) for r in stats]))
                                                 for room in ['d1', 'h1', 'i1', 'j1', 'u1', 'v1']},
                   seed_table_platform_scores={r['seed']: float(r['platform_score']) for r in stats if r['platform_score']},
                   score_table_rows=len(scores), original_config=original['config'],
                   limitations=['No hidden ground truth or platform re-evaluation.',
                                'U prediction distributions not independently recomputed.',
                                'Cannot verify identity to absent historical platform uploads.',
                                'YAML is a recipe, not full reproduction source/data-order/environment evidence.',
                                'ONNX structural validation and strict checkpoint load do not prove numerical equivalence.'])
    save(OUT / 'files.json', entries)
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def smoke():
    """Raw-input numerical check, not an independent accuracy estimate."""
    import blosc2
    import onnxruntime as ort
    from utils import window
    torch.set_num_threads(4)
    rows = read_csv(ROOT / 'manifests/train_windows.csv')
    chosen = []
    for label in range(5):
        candidates = [r for r in rows if int(r['headcount']) == label]
        first = candidates[0]
        second = next(r for r in candidates if r['identifier'] != first['identifier'])
        chosen.extend([first, second])
    samples = []
    for r in chosen:
        folder = Path('D:/huawei2026-data/labeled')
        csi = blosc2.load_array(str(folder / f'CSI_{r["identifier"]}.b2'))
        ati = blosc2.load_array(str(folder / f'ATI_{r["identifier"]}.b2'))
        samples.append(window(csi, ati, r))
        del csi, ati
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    results = []
    for seed in [0, 42]:
        checkpoint = torch.load(PACKAGE / f'weights/checkpoints/base037_full_s{seed}.pt',
                                map_location='cpu', weights_only=True)
        model = StructuredModel('spectral', 'physical').eval()
        model.load_state_dict(checkpoint['model_state'], strict=True)
        session = ort.InferenceSession(str(PACKAGE / f'weights/onnx/base037_full_s{seed}.onnx'),
                                       sess_options=options, providers=['CPUExecutionProvider'])
        windows = []
        for row, (x, dt) in zip(chosen, samples):
            z = session.run(None, {'frames': x, 'intervals': dt})[0]
            with torch.inference_mode():
                t = model(torch.from_numpy(x)[None], torch.from_numpy(dt)[None])[0].numpy()
            assert np.isfinite(z).all() and np.isfinite(t).all()
            p = np.exp(z - z.max()); p /= p.sum()
            q = np.exp(t - t.max()); q /= q.sum()
            windows.append(dict(identifier=row['identifier'], start=int(row['start']), label=int(row['headcount']),
                                onnx_prediction=int(z.argmax()), torch_prediction=int(t.argmax()),
                                max_probability_difference=float(np.abs(p-q).max())))
            print('RAW_CHECK', seed, row['identifier'], windows[-1]['max_probability_difference'], flush=True)
        results.append(dict(seed=seed, windows=windows,
                            same_argmax=sum(r['onnx_prediction'] == r['torch_prediction'] for r in windows),
                            max_probability_difference=max(r['max_probability_difference'] for r in windows)))
        del session, model, checkpoint
    save(OUT / 'raw_smoke.json', dict(selection='First two distinct records per label, first manifest windows; all training-source data.',
                                    independent_accuracy=False, seeds=results))


if __name__ == '__main__':
    smoke() if '--smoke' in sys.argv else main()
