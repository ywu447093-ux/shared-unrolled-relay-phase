"""Enrich the completed recent baseline with radian errors and synchronized timing."""
import json,time
from collections import defaultdict
import numpy as np,torch
import suite as s
import reviewer_validation as v
from fouriergs_paired import PairedFourierGSNet

OUT=s.ROOT/'work/fouriergs_v1';F=OUT/'learning/fouriergs_paired'

def main():
    assert (OUT/'all_complete.json').exists()
    torch.set_num_threads(4);data,split,records,_=s.load_data(False)
    prediction=torch.load(F/'predictions.pt',map_location='cpu',weights_only=False)
    rows=json.loads((F/'holdout_metrics.json').read_text())
    assert prediction['names']==[r['name'] for r in rows]
    pred=prediction['predictions'];truth=data[7][split['holdout']]
    for i,row in enumerate(rows):row['metrics'].update(v.phase_metrics(pred[i:i+1,1:2],truth[i:i+1,1:2]))
    groups=defaultdict(list)
    for row in rows:groups[(row['cell_type'],row['content_id'])].append(row)
    assert len(groups)==64
    means={key:{k:float(np.mean([x['metrics'][k] for x in values])) for k in rows[0]['metrics']} for key,values in groups.items()}
    rng=np.random.default_rng(20261104);cats=sorted({k[0] for k in means});bootstrap={k:np.zeros(2000) for k in rows[0]['metrics']}
    macro={k:0. for k in rows[0]['metrics']}
    for cat in cats:
        selected=[x for (c,cid),x in means.items() if c==cat];ix=rng.integers(len(selected),size=(2000,len(selected)))
        for k in macro:
            x=np.array([v[k] for v in selected]);macro[k]+=float(x.mean())/len(cats);bootstrap[k]+=x[ix].mean(1)/len(cats)
    s.dump(F/'holdout_metrics_enriched.json',rows)
    model=PairedFourierGSNet().cuda().eval();ck=torch.load(F/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    b=[t[split['holdout'][:1]].cuda() for t in data];times=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(35):
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b).float()
            torch.cuda.synchronize()
            if i>=5:times.append((time.perf_counter()-start)*1000)
    assert torch.isfinite(p).all()
    complete=json.loads((F/'complete.json').read_text())
    summary={'macro':macro,'ci95':{k:np.quantile(x,[.025,.975]).tolist() for k,x in bootstrap.items()},
       'parameters':sum(p.numel() for p in model.parameters()),'selected_epoch':complete['best_epoch'],
       'training_elapsed_s':complete['elapsed_s'],'latency_ms':{'median':float(np.median(times)),'p10':float(np.quantile(times,.1)),'p90':float(np.quantile(times,.9)),'samples':times},
       'peak_inference_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint_sha256':s.sha(F/'best.pt')}
    s.dump(OUT/'internal_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
