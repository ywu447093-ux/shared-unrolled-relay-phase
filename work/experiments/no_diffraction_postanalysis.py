"""CPU-only internal metric enrichment for the matched measurement ablation."""
import json
from collections import defaultdict
import numpy as np,torch
import suite as s
import reviewer_validation as v

OUT=s.ROOT/'work/no_diffraction_v1';F=OUT/'learning/stage4_no_diffraction'

def main():
    assert (OUT/'all_complete.json').exists();torch.set_num_threads(2)
    data,split,records,_=s.load_data(False);p=torch.load(F/'predictions.pt',weights_only=False);rows=json.loads((F/'holdout_metrics.json').read_text());truth=data[7][split['holdout']]
    assert p['names']==[r['name'] for r in rows]
    for i,r in enumerate(rows):r['metrics'].update(v.phase_metrics(p['predictions'][i:i+1,1:2],truth[i:i+1,1:2]))
    groups=defaultdict(list)
    for r in rows:groups[(r['cell_type'],r['content_id'])].append(r)
    assert len(groups)==64
    means={key:{k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']} for key,rr in groups.items()}
    keys=rows[0]['metrics'];macro={k:0. for k in keys};draws={k:np.zeros(2000) for k in keys};rng=np.random.default_rng(20261104)
    for cat in sorted({k[0] for k in means}):
        rr=[x for (c,cid),x in means.items() if c==cat];ix=rng.integers(len(rr),size=(2000,len(rr)))
        for k in keys:
            x=np.array([r[k] for r in rr]);macro[k]+=float(x.mean())/5;draws[k]+=x[ix].mean(1)/5
    s.dump(OUT/'internal_summary.json',{'macro':macro,'ci95':{k:np.quantile(x,[.025,.975]).tolist() for k,x in draws.items()},'selected_epoch':json.loads((F/'complete.json').read_text())['best_epoch'],'parameters_retained':1396998,'inactive_consistency_logits':4})
    s.dump(F/'holdout_metrics_enriched.json',rows);print(macro,flush=True)

if __name__=='__main__':main()
