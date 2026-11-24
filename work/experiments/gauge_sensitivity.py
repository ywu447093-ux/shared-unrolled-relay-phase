"""Read-only sensitivity to a common observed-background phase convention."""
import json
from collections import defaultdict
import numpy as np,torch
import suite as s
import reviewer_validation as v

def main():
    torch.set_num_threads(2);cache=torch.load(v.OUT/'secondary_cache.pt',map_location='cpu',weights_only=False)
    data=cache['data'];records=cache['records'];truth=data[7][:,1].numpy().astype(np.float64);mask=truth>np.maximum(.05*truth.max((-2,-1),keepdims=True),.03)
    support=data[5][:,0].numpy().astype(np.float64);flat=data[4][:,0].numpy().astype(np.float64)
    illumination=flat>=.12*flat.max((-2,-1),keepdims=True);weight=(1-support)**4*illumination
    border=np.zeros((128,128));border[:4]=border[-4:]=1;border[:,:4]=border[:,-4:]=1
    for i in range(210):
        if weight[i].sum()<1e-8:weight[i]=illumination[i]*border
        if weight[i].sum()<1e-8:weight[i]=border
    methods=[*v.NEURAL,*v.CLASSICAL['secondary']]
    protocol={'status':'read-only diagnostic fixed before its metric computation','scope':'Retrospective105-source cohort; no training or primary output modification',
      'gauge':'Common circular phase mean over (1-support)^4 with observed flat above12% maximum; illuminated4pixelborder then gridborder fallback',
      'output':'Subtract observed-background circular mean, wrap to[-pi,pi], clip to[0,pi]',
      'reference_usage':'Reference phase only evaluates errors, never chooses offset or method',
      'code_sha256':s.sha(__file__),'methods':methods}
    out=s.ROOT/'work/evidence_analysis';s.dump(out/'gauge_sensitivity_protocol.json',protocol);result={};perimage=[]
    for method in methods:
        pred=torch.load(v.OUT/'secondary'/method/'predictions.pt',map_location='cpu',weights_only=False)['predictions'][:,1].numpy().astype(np.float64)
        offset=np.arctan2((np.sin(pred)*weight).sum((-2,-1)),(np.cos(pred)*weight).sum((-2,-1)))
        aligned=np.clip((pred-offset[:,None,None]+np.pi)%(2*np.pi)-np.pi,0,np.pi)
        fields={'raw_rmse_rad':np.sqrt(((pred-truth)**2*mask).sum((-2,-1))/mask.sum((-2,-1))),
          'observed_background_rmse_rad':np.sqrt(((aligned-truth)**2*mask).sum((-2,-1))/mask.sum((-2,-1))),
          'observed_background_offset_rad':offset}
        means={k:float(np.mean([np.mean(x[[i for i,r in enumerate(records) if r['cell_type']==c]]) for c in sorted({r['cell_type'] for r in records})])) for k,x in fields.items()}
        result[method]=means
        perimage.extend({'method':method,'name':r['name'],'content_id':r['content_id'],'category':r['cell_type'],**{k:float(x[i]) for k,x in fields.items()}} for i,r in enumerate(records))
    s.dump(out/'gauge_sensitivity.json',{'protocol':protocol,'macro':result,'per_observation':perimage})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
