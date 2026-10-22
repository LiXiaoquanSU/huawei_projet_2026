from pathlib import Path
import deliver
from query_network import build
deliver.WORK=Path('D:/huawei2026-experiments/query_20261021')
deliver.build=build
if __name__=='__main__':deliver.main()
