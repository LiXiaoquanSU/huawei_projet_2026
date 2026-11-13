"""Reuse full raw-window audit with new head and exact scored054 control."""
from ra_common import *
from export_routes import RouteHead,NAMES

def main():
    # Audit procedure is versioned in repository; changes below only bind the
    # experiment/head/control, not thresholds or number of tested raw windows.
    source=(ROOT/'experiments/next053_20261025/audit.py').read_text(encoding='utf-8')
    source=source.replace('from n53_common import *','from ra_common import *')
    source=source.replace('from export_batch import FeatureHead,NAMES','from export_routes import RouteHead,NAMES')
    source=source.replace("WORK/'preprocessing.onnx'","PREV/'preprocessing.onnx'")
    source=source.replace("'__n53_control'","'__ra_control'")
    source=source.replace("FeatureHead(selection['method'])","RouteHead(selection)")
    source=source.replace("WORK/'rows.json'","PREV/'rows.json'")
    source=source.replace("if selection['method']=='transfer_low_guard' and base.argmax()>=2:","if base.argmax()>=2:")
    source=source.replace('Original054-style control','Scored selective054 control')
    source=source.replace('N53_RAW','RA_RAW').replace('N53_VALIDATED','RA_VALIDATED')
    exec(compile(source,str(Path(__file__)), 'exec'),{'__name__':'__main__'})
if __name__=='__main__':main()
