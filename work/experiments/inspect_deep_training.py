"""CPU diagnostic of saved deep-model states on fixed training observations.

This does not alter a model, its training schedule or validation selection.
"""
import argparse,json,math
import torch
import suite as s

ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['stage6','stage8'],required=True)
ap.add_argument('--assets-root')
args=ap.parse_args();torch.set_num_threads(4)
if args.assets_root:s.OLD=s.Path(args.assets_root).resolve()
folder=s.OUT/'learning'/args.arm
data,split,records,_=s.load_data(False)
result={'arm':args.arm,'scope':'CPU FP32 diagnostic on first four training observations; not training precision or an evaluation comparison','checkpoints':{}}
for name in ['best.pt','resume.pt']:
 ck=torch.load(folder/name,map_location='cpu',weights_only=False)
 m=s.model_for(args.arm).eval();m.load_state_dict(ck['model'])
 rows=[]
 for idx in split['train'][:4]:
  b=[t[idx:idx+1] for t in data];m.zero_grad(set_to_none=True)
  pred=m(b);loss=s.objective(pred,b,s.DeterministicPhysicalLoss());loss.backward()
  g2=sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)
  phase=pred[:,1:2]
  rows.append({'observation_index':idx,'phase_mean':float(phase.detach().mean()),'phase_min':float(phase.detach().min()),
   'phase_max':float(phase.detach().max()),'phase_at_zero_fraction':float((phase.detach()==0).float().mean()),
   'phase_at_pi_fraction':float((phase.detach()==math.pi).float().mean()),'loss':float(loss.detach()),'parameter_gradient_norm':math.sqrt(g2)})
 result['checkpoints'][name]={'epoch':ck['epoch'],'observations':rows}
 del m,ck
s.dump(folder/'training_state_diagnostic.json',result)
print(json.dumps(result,indent=2))
