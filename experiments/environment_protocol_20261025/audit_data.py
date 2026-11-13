"""Fresh ATI audit of all198 records, limited raw-CSI screening with explicit scope."""
import csv,gc,hashlib,json,sys
from pathlib import Path
import blosc2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/environment_protocol_20261025')

def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def hashed(identifier):return hashlib.sha256(identifier.encode()).hexdigest()

def main():
    out=[];rawcases=[]
    for part in ['labeled','unlabeled']:
        folder=Path('D:/huawei2026-data')/part;records=readcsv(folder/'samples.csv')
        sample_ids=set()
        if part=='unlabeled':
            for room in sorted({r['room'] for r in records if r['room']!='unknown'}):
                sample_ids.add(min((r['identifier'] for r in records if r['room']==room),key=hashed))
            sample_ids.update(r['identifier'] for r in records if r['room']=='unknown')
        for row in records:
            identifier=row['identifier'];ati=blosc2.load_array(str(folder/f'ATI_{identifier}.b2')).reshape(-1)
            assert ati.dtype==np.float32 and len(ati)==12000 and np.isfinite(ati).all()
            dt=ati[:-1];positive=dt[dt>0]
            out.append(dict(split=part,identifier=identifier,room=row['room'],layout=row['layout'],frames=len(ati),
              median_interval_us=float(np.median(positive)),p95_interval_us=float(np.percentile(positive,95)),
              max_interval_us=float(positive.max()),duration_seconds=float(dt.sum(dtype=np.float64)/1e6),
              gaps_gt20ms=int((dt>20000).sum()),nonpositive_intervals=int((dt<=0).sum()),
              negative_intervals=int((dt<0).sum()),last_padding=float(ati[-1]),
              note='Fresh ATI scan; no headcount labels used.'))
            if identifier not in sample_ids:continue
            csi=blosc2.load_array(str(folder/f'CSI_{identifier}.b2'));assert csi.shape==(12000,1,3,1024) and csi.dtype==np.complex64
            x=csi[:600,0];power=x.real.astype(np.float64)**2+x.imag.astype(np.float64)**2
            logamp=.5*np.log1p(power)
            rawcases.append(dict(identifier=identifier,room=row['room'],layout=row['layout'],shape=list(csi.shape),dtype=str(csi.dtype),
              inspected_frames=600,finite=bool(np.isfinite(x).all()),active_fraction=float((power>1e-8).mean()),
              per_rx_log_amplitude_mean=logamp.mean((0,2)).tolist(),
              per_rx_temporal_variation=logamp.std(0).mean(-1).tolist()))
            del csi,x,power,logamp;gc.collect()
    with (WORK/'fresh_ati_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    summary=dict(fresh_ati_records=len(out),raw_csi_screen_records=len(rawcases),raw_scope='First600frames, one deterministic record per knownUroom plus all5unknownrecords; not a fullCSIaudit.',
      splits={part:dict(records=sum(r['split']==part for r in out),
        median_record_intervals_us=[min(r['median_interval_us'] for r in out if r['split']==part),max(r['median_interval_us'] for r in out if r['split']==part)],
        trace_durations_seconds=[min(r['duration_seconds'] for r in out if r['split']==part),max(r['duration_seconds'] for r in out if r['split']==part)],
        negative_interval_count=sum(r['negative_intervals'] for r in out if r['split']==part),
        records_with_20ms_gaps=sum(r['gaps_gt20ms']>0 for r in out if r['split']==part)) for part in ['labeled','unlabeled']},raw_cases=rawcases,
      inference='Differences indicate sampling/background/occupancy mixtures, not measured counting accuracy. Unknown records not treated as one room.')
    (WORK/'data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='raw_cases'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
