"""Equal-observation GS/HIO/HIO-ER/NCPP exploration with a transmitted background.

Methods are explicitly adapted to noisy amplitude, bounded phase and saturation.
The soft exterior constraint acts on contrast relative to flat-field illumination.
No reconstructed prediction is selected using evaluation targets.
"""
import json,math,time
from pathlib import Path
import numpy as np
import torch
import suite as s

METHODS=['GS','HIO','HIO_ER','NCPP']

@torch.inference_mode()
def reconstruct(b,method,cfg,keep_trace=False):
 obs,pn,pp,a,flat,sup,sat=b[:7];prop,inv=s.make_propagators(pp,128)
 mu=(obs*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/s._camera_modulus_gain(pp)
 strength=cfg.get('strength',0.);gate=1-strength*(1-sup);bg=flat.to(torch.complex64)
 # Compare three prespecified starts on validation. One fixed phase pattern is
 # reused across observations, so changing batching cannot change the start.
 init=cfg.get('initialization','zero')
 rng=torch.Generator(device=a.device);rng.manual_seed(101)
 phase=torch.rand(a[:1].shape,generator=rng,device=a.device).expand_as(a)
 phase=phase*(0. if init=='zero' else .1 if init=='small_random' else math.pi)
 field=torch.polar(a.clamp(0,1.25),phase);traces=[];steps=0
 def modulus_projection(u,target):
  v=prop(u);mod=v.abs();bound=torch.where(sat>.5,torch.maximum(mod,mu),target)
  return inv(bound*v/(mod+1e-8))
 def object_projection(u,localized=True):
  if method=='GS':amp=a.clamp(0,1.25)
  else:
   contextual=(flat+gate*(u.abs().clamp(0,1.25)-flat)).clamp(0,1.25)
   amp=(.9*a+.1*contextual).clamp(0,1.25)
  phase=torch.angle(u).clamp(0,math.pi)
  return torch.polar(amp,gate*phase if localized else phase)
 def hybrid_projection(previous,candidate,beta):
  # Apply the paired amplitude/phase object constraint inside the specimen,
  # while retaining the HIO feedback of background-relative contrast outside.
  accepted=object_projection(candidate,localized=False)
  return bg+gate*(accepted-bg)+(1-gate)*((previous-bg)-beta*(candidate-bg))
 def trace():
  if keep_trace:
   pred=torch.cat([field.abs(),torch.angle(field).clamp(0,math.pi)],1)
   values=[]
   for j in range(len(obs)):
    metrics=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1])
    values.append(dict(zip(s.METRIC_NAMES,map(float,metrics))))
   traces.append({'propagation_pairs':steps,'metrics':values})
 if method=='NCPP':
  gammas=[1.2,1.,.8];betas=np.linspace(3.,1.,11)
  for gamma in gammas:
   target=(obs.clamp(0,1).pow(gamma)*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/s._camera_modulus_gain(pp)
   for beta in betas:
    for inner in range(cfg['inner']):
     candidate=modulus_projection(field,target);steps+=1
     field=hybrid_projection(field,candidate,float(beta))
     field=object_projection(modulus_projection(field,target));steps+=1
    if keep_trace:trace()
 else:
  total=cfg['iterations'];beta=cfg.get('beta',.9)
  for iteration in range(total):
   candidate=modulus_projection(field,mu);steps+=1
   if method=='GS':field=object_projection(candidate)
   elif method=='HIO':
    field=hybrid_projection(field,candidate,beta)
   else:
    if iteration%20<15:field=hybrid_projection(field,candidate,beta)
    else:field=object_projection(candidate)
   if keep_trace and (iteration+1 in [1,2,4,8,16,32,64,100,128,256,300,512,1000]):
    raw=field;field=object_projection(field);trace();field=raw
  field=object_projection(field)
 assert torch.isfinite(field).all(),(method,cfg)
 return torch.cat([field.abs(),torch.angle(field).clamp(0,math.pi)],1),traces

def evaluate(data,ids,records,method,cfg,keep=False):
 rows=[];preds=[];seconds=0.;trace_batches=[]
 for off in range(0,len(ids),16):
  b=[t[ids[off:off+16]].cuda() for t in data];torch.cuda.synchronize();start=time.perf_counter()
  pred,traces=reconstruct(b,method,cfg,False);torch.cuda.synchronize();seconds+=time.perf_counter()-start
  for j,row in enumerate(records[off:off+16]):
   values=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1])
   rows.append({**row,'metrics':dict(zip(s.METRIC_NAMES,map(float,values)))})
  if keep:preds.append(pred.cpu())
 return s.macro(rows),rows,torch.cat(preds) if keep else None,seconds

def main():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 design=s.OUT/'classical_design_at_execution.json'
 if not design.exists():s.dump(design,{'time_unix':time.time(),'stage':'before the first classical validation candidate',
   'classical_sha256':s.sha(__file__),'protocol_sha256':s.sha(s.HERE/'PROTOCOL.md'),
   'paired_constraints':'amplitude-aware accepted-region projection; background-relative exterior feedback; saturation lower bounds',
   'supplementary_training':['full-calibration image baselines','separate lower-learning-rate depth sensitivity','per-observation PhysenNet']})
 else:assert json.loads(design.read_text())['classical_sha256']==s.sha(__file__)
 data,split,records,_=s.load_data(False)
 directory=s.OUT/'classical';directory.mkdir(exist_ok=True)
 # Same sampled propagation used in both the classical and neural implementations.
 b=[t[split['validation'][:2]].cuda() for t in data];prop,inv=s.make_propagators(b[2],128)
 u=torch.polar(b[7][:,:1],b[7][:,1:2]);relative=float((inv(prop(u))-u).norm()/u.norm())
 assert relative<1e-5
 s.dump(directory/'propagation_check.json',{'relative_roundtrip_error':relative})
 candidates={
  'GS':[{'strength':w,'iterations':n} for w in [0.,.65,1.] for n in [100,300,1000]],
  'HIO':[{'strength':w,'beta':beta,'iterations':n} for w in [.35,.65,1.] for beta in [.5,.9] for n in [100,300,1000]],
  'HIO_ER':[{'strength':w,'beta':beta,'iterations':n} for w in [.35,.65,1.] for beta in [.5,.9] for n in [100,300,1000]],
  'NCPP':[{'strength':w,'inner':n} for w in [.35,.65,1.] for n in [10,31]]}
 candidates={k:[{**cfg,'initialization':init} for cfg in values for init in ['zero','small_random','uniform_random']] for k,values in candidates.items()}
 for method in METHODS:
  target=directory/method;target.mkdir(exist_ok=True)
  if (target/'complete.json').exists():continue
  tuning=[]
  for ix,cfg in enumerate(candidates[method]):
   path=target/f'validation_{ix:02d}.json'
   if path.exists():entry=json.loads(path.read_text())
   else:
    macro,rows,_,seconds=evaluate(data,split['validation'],records['validation'],method,cfg)
    entry={'config':cfg,'macro':macro,'score':s.score(macro),'seconds':seconds,'per_sample':rows};s.dump(path,entry)
   tuning.append(entry);print(method,cfg,entry['macro']['phase_foreground_ssim'],flush=True)
  best=max(tuning,key=lambda x:x['score'])
  s.dump(target/'selection.json',{'selected':best['config'],'validation_score':best['score'],'rule':'highest fixed validation composite; no evaluation target stopping or alignment'})
  macro,rows,preds,seconds=evaluate(data,split['holdout'],records['holdout'],method,best['config'],True)
  s.save(target/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':preds})
  s.dump(target/'holdout_metrics.json',rows)
  # A fixed first example of each category gives a diagnostic trajectory without target selection.
  categories=sorted({r['cell_type'] for r in records['holdout']});chosen=[next(i for i,r in enumerate(records['holdout']) if r['cell_type']==c) for c in categories]
  bb=[t[[split['holdout'][j] for j in chosen]].cuda() for t in data];_,trace=reconstruct(bb,method,best['config'],True)
  s.dump(target/'trajectory.json',{'names':[records['holdout'][j]['name'] for j in chosen],'categories':categories,'records':trace})
  # Synchronised single-frame timings include the selected fixed iteration count.
  single=[t[:1] for t in bb];timings=[]
  for repeat in range(7):
   torch.cuda.synchronize();start=time.perf_counter();reconstruct(single,method,best['config']);torch.cuda.synchronize()
   if repeat>=2:timings.append((time.perf_counter()-start)*1000)
  s.dump(target/'complete.json',{'status':'complete','config':best['config'],'macro':macro,'evaluation_s':seconds,
    'latency_ms':{'median':float(np.median(timings)),'p10':float(np.quantile(timings,.1)),'p90':float(np.quantile(timings,.9))},
    'method_note':'paired-observation adaptation with nonzero transmitted background; not unchanged published algorithm',
    'script_sha256':s.sha(__file__)})
 # Primary direct baselines also receive the complete calibrated modulus.
 import calibrated
 calibrated.main()
 # Keep a learning-rate check separate from the frozen matched-depth arms.
 import stability
 stability.main()
 # A second published deep-learning family fits each observation without labels.
 import physennet
 physennet.main()
if __name__=='__main__':main()
