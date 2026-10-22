"""Use the common exporter with the ratio model factory."""
import torch
import export_candidates as ex
from ratio import fresh_ratio
from run import save,WORK

if __name__=='__main__':
    torch.set_num_threads(4)
    ex.fresh=lambda variant,seed:fresh_ratio(seed)
    result=ex.export('ratio',42)
    save(WORK/'exports/ratio_registry.json',[result])
