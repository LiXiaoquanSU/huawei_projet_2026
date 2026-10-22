"""Sequential screen, replication, final training and resource-checked export."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    for stage in ('screen','replicate','full'):
        subprocess.run([sys.executable,'-X','utf8',str(HERE/'train.py'),'--stage',stage],check=True)
    subprocess.run([sys.executable,'-X','utf8',str(HERE/'export.py')],check=True)
    print('ALL_TRAINING_EXPORT_COMPLETE',flush=True)
