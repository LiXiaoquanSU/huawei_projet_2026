"""Continue the registered run sequentially; each completed fit is resumable."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    for stage in ('replicate','full'):
        subprocess.run([sys.executable,'-X','utf8',str(HERE/'run.py'),'--stage',stage],check=True)
    subprocess.run([sys.executable,'-X','utf8',str(HERE/'export.py')],check=True)
    print('TRAINING_AND_EXPORT_COMPLETE',flush=True)
