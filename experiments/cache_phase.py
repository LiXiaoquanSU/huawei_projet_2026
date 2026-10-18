"""Cache deterministic, per-window relative CSI features for known rooms only."""
import hashlib
from pathlib import Path
import torch
from phase_models import PhaseModel
from train_baseline import read_csv,features

torch.set_num_threads(4)
out=Path('artifacts/phase_cache');out.mkdir(parents=True,exist_ok=True)
base=Path('artifacts/baseline_a1_seed42')
model=PhaseModel().cuda().eval()
identity=hashlib.sha256(Path('phase_models.py').read_bytes()+Path('improved_models.py').read_bytes()).hexdigest()
for split in ['train','val']:
    manifest=base/f'{split}_windows.csv'
    features(read_csv(manifest),model,out/f'{split}_features.pt',identity+hashlib.sha256(manifest.read_bytes()).hexdigest())
