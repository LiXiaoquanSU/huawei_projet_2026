"""Full raw-window audit, bind dual decoder head and current054 control."""
from decoder_round import *

def main():
    source=(ROOT/'experiments/next053_20261025/audit.py').read_text(encoding='utf-8')
    source=source.replace('from n53_common import *','from decoder_round import *\nWORK=DEC')
    source=source.replace('from export_batch import FeatureHead,NAMES','from decoder_export import DecoderHead,NAMES')
    source=source.replace("WORK/'preprocessing.onnx'","PREV/'preprocessing.onnx'").replace("'__n53_control'","'__ra_control'")
    source=source.replace("FeatureHead(selection['method'])","DecoderHead(selection)").replace("WORK/'rows.json'","PREV/'rows.json'")
    source=source.replace("if selection['method']=='transfer_low_guard' and base.argmax()>=2:","if selection['scope']=='low' and base.argmax()>=2:")
    source=source.replace('Original054-style control','Scored selective054 control').replace('N53_RAW','DEC_RAW').replace('N53_VALIDATED','DEC_VALIDATED')
    exec(compile(source,str(Path(__file__)), 'exec'),{'__name__':'__main__'})
if __name__=='__main__':main()
