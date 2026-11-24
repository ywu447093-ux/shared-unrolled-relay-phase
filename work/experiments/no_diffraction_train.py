"""Identical shared correction network with diffraction removed from map and loss."""
import argparse,json,time,math
from types import SimpleNamespace
import torch
import suite as s
import reviewer_validation as v
from fouriergs_train import evaluate_secondary

OUT=s.ROOT/'work/no_diffraction_v1';ARM='stage4_no_diffraction'

class NoDiffraction(s.Reconstructor):
    def __init__(self):super().__init__('stage4')
    def forward(self,b):
        pn,a,flat,sup=b[1],b[3].clamp(0,1.25),b[4],b[5]
        phase=torch.zeros_like(a);amp=a;fl=flat.clamp_min(1e-4)
        absorption=((fl-a)/fl).clamp(-.1,.15)*20;maps=pn[:,:3,None,None].expand(-1,-1,128,128)
        zero=torch.zeros_like(a)
        for k in range(4):
            features=torch.cat([phase/math.pi,zero,zero,absorption,sup.clamp(0,1),flat,maps],1)
            delta=self.prior(features,k/3,k).float()
            phase=(phase+.5*math.pi*torch.tanh(delta[:,1:2])).clamp(0,math.pi)
            amp=(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25)
        return torch.cat([amp,phase],1)

def evaluate_all():
    import numpy as np
    ck=torch.load(OUT/'learning'/ARM/'best.pt',map_location='cpu',weights_only=False)
    model=NoDiffraction().cuda().eval();model.load_state_dict(ck['model'])
    for condition in v.CONDITIONS:
        folder=OUT/'evaluation'/condition/ARM;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():continue
        cache=torch.load(v.OUT/(condition+'_cache.pt'),weights_only=False);rows=[];preds=[]
        with torch.inference_mode():
            for off in range(0,210,4):
                b=[t[off:off+4].cuda() for t in cache['data']]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
                for j,r in enumerate(cache['records'][off:off+4]):
                    vv=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1]);mm=dict(zip(s.METRIC_NAMES,map(float,vv)))
                    mm.update(v.phase_metrics(pred[j:j+1,1:2],b[7][j:j+1,1:2]));rows.append({**r,'metrics':mm})
                preds.append(pred.cpu())
        macro={k:float(np.mean([np.mean([r['metrics'][k] for r in rows if r['cell_type']==c]) for c in sorted({r['cell_type'] for r in rows})])) for k in rows[0]['metrics']}
        s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':torch.cat(preds)})
        s.dump(folder/'per_observation_metrics.json',rows);s.dump(folder/'complete.json',{'status':'complete','macro':macro,'observations':210,'source_contents':105})
        print('NO_DIFFRACTION',condition,macro['phase_foreground_rmse_rad'],flush=True)

def main():
    OUT.mkdir(exist_ok=True)
    cfg={'purpose':'Isolate diffraction information from the capacity and depth of the shared correction network',
      'timing':'Frozen before fitting or inspecting no-diffraction control scores',
      'architecture':'Same four-stage shared network, attention, stage coordinate, amplitude head and phase increment/clipping; two residual quadratures set to zero; physical phase increment removed. Four retained consistency logits are inactive.',
      'loss':'Original supervised amplitude and phase terms; physical residual omitted to exclude diffraction from objective',
      'budget':'seed101, random initialization, original 40epochs and original1184observations/epoch, batch8, LR3e-4, same augmentation draws and validation selection',
      'invariance':'Forward and objective never use SPS, physical calibration/extrema or saturation flags; optical-geometry maps are retained',
      'code_sha256':s.sha(__file__),'suite_sha256':s.sha(s.__file__)}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:s.dump(path,cfg)
    print('QUEUED: no-diffraction control waits for FourierGSNet completion',flush=True)
    while not (s.ROOT/'work/fouriergs_v1/all_complete.json').exists():time.sleep(15)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    s.seed(101);probe=NoDiffraction().cuda().eval();data,split,records,groups=s.load_data();b=[t[split['train'][:2]].cuda() for t in data];changed=[t.clone() for t in b]
    changed[0]=torch.rand_like(changed[0]);changed[2]=torch.rand_like(changed[2]);changed[6]=1-changed[6]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.equal(probe(b),probe(changed))
    del probe,b,changed;torch.cuda.empty_cache()
    original=s.model_for;s.model_for=lambda arm:NoDiffraction() if arm==ARM else original(arm)
    original_objective=s.objective;s.objective=lambda pred,b,physics,amplitude_only=False:original_objective(pred,b,physics,True)
    s.LABELS[ARM]='Four-stage shared correction without diffraction';s.OUT=OUT
    s.train(ARM,SimpleNamespace(seed=101,epochs=40,lr=3e-4,microbatch=8),data,split,records,groups)
    del data,groups;evaluate_all();s.dump(OUT/'all_complete.json',{'status':'complete','arm':ARM,'forward_diffraction_invariance':True})

if __name__=='__main__':main()
