"""Per-observation PhysenNet adaptation; no supervised pretraining or target loss.

The 32/64/128/256/512 U-Net and normalization/activation pattern follow the
authors' TensorFlow source. This PyTorch port changes propagation to the paired
relay-screen operator, anchors amplitude to Am, bounds phase to [0, pi], and fits
the same robust censored measurement loss used in this study. It is reported as
an adaptation, never as the original paper's experiment or published accuracy.
"""
import json,math,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import suite as s

class Conv(nn.Module):
 def __init__(self,cin,cout,first_relu=False,transpose=False,stride=1,last=False):
  super().__init__()
  op=nn.ConvTranspose2d(cin,cout,3,stride=stride,padding=1,output_padding=1 if stride==2 else 0,bias=False) if transpose else nn.Conv2d(cin,cout,3,padding=1,bias=False)
  self.net=nn.Sequential(op,nn.BatchNorm2d(cout,eps=.001,momentum=.01),nn.ReLU() if first_relu or last else nn.LeakyReLU(.2))
  nn.init.trunc_normal_(op.weight,mean=0.,std=.1,a=-.2,b=.2)
 def forward(self,x):return self.net(x)

class PhysenUNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.enc=nn.ModuleList([nn.Sequential(Conv(1,32,True,True),Conv(32,32)),
   nn.Sequential(Conv(32,64),Conv(64,64)),nn.Sequential(Conv(64,128,True),Conv(128,128)),
   nn.Sequential(Conv(128,256),Conv(256,256)),nn.Sequential(Conv(256,512),Conv(512,512))])
  self.up=nn.ModuleList([Conv(512,256,transpose=True,stride=2),Conv(256,128,transpose=True,stride=2),
   Conv(128,64,transpose=True,stride=2),Conv(64,32,transpose=True,stride=2)])
  self.dec=nn.ModuleList([nn.Sequential(Conv(512,256),Conv(256,256)),nn.Sequential(Conv(256,128),Conv(128,128)),
   nn.Sequential(Conv(128,64),Conv(64,64)),nn.Sequential(Conv(64,32),Conv(32,32))])
  self.out=Conv(32,1,last=True)
 def forward(self,x):
  feats=[]
  for i,layer in enumerate(self.enc):
   x=layer(x if i==0 else F.max_pool2d(x,2));feats.append(x)
  for i,(up,dec) in enumerate(zip(self.up,self.dec)):x=dec(torch.cat([feats[3-i],up(x)],1))
  return self.out(x).float()

def referenced_phase(raw,b):
 """Fix the unobservable global phase using observed background, not labels."""
 flat=b[4];valid=(flat>=.12*flat.amax((-2,-1),keepdim=True)).float()
 weights=(1-b[5].clamp(0,1)).pow(4)*valid
 if float(weights.sum())<1e-4:
  border=torch.zeros_like(flat);border[...,:4,:]=1;border[...,-4:,:]=1;border[...,:,:4]=1;border[...,:,-4:]=1
  weights=border*valid
  if float(weights.sum())<1e-4:weights=border
 phasor=(weights*torch.polar(torch.ones_like(raw),raw)).sum((-2,-1),keepdim=True)
 reference=torch.angle(phasor)
 relative=torch.atan2(torch.sin(raw-reference),torch.cos(raw-reference))
 return relative.clamp(0,math.pi)

def fit(b,steps=(100,300)):
 s.seed(101);m=PhysenUNet().cuda().train();opt=torch.optim.Adam(m.parameters(),lr=1e-3)
 physics=s.DeterministicPhysicalLoss().cuda();preds={};torch.cuda.synchronize();start=time.perf_counter()
 inp=b[0].square();a=b[3].clamp(0,1.25)
 for step in range(1,max(steps)+1):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):phase=referenced_phase(m(inp),b)
  loss=physics(a,phase,b[0],b[2],saturation_mask=b[6]);assert torch.isfinite(loss)
  loss.backward();opt.step()
  if step in steps:
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):phase=referenced_phase(m(inp),b)
   preds[step]=torch.cat([a,phase],1).detach().cpu()
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 del m,opt,physics;torch.cuda.empty_cache();return preds,seconds

def main():
 torch.set_num_threads(4);data,split,records,_=s.load_data(False)
 folder=s.OUT/'classical/PhysenNet';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 s.dump(folder/'phase_reference_design.json',{'stage':'before the first per-observation fit',
  'reference':'circular mean of raw phase in amplitude-derived illuminated background, weights (1-support)^4; illuminated-border fallback',
  'reason':'fix global phase using the same known-background convention as the other paired-observation algorithms',
  'targets':'no reference phase used','script_sha256':s.sha(__file__)})
 candidates=[100,300]
 validation_rows={n:[] for n in candidates}
 for j,idx in enumerate(split['validation']):
  cache=folder/'validation'/f'{j:03d}.pt'
  if cache.exists():obj=torch.load(cache,map_location='cpu',weights_only=True)
  else:
   b=[t[idx:idx+1].cuda() for t in data];preds,seconds=fit(b,candidates)
   obj={'predictions':preds,'seconds':seconds};cache.parent.mkdir(exist_ok=True);s.save(cache,obj)
  for n in candidates:
   pred=obj['predictions'][n]
   vals=s.compute_validation_metrics(pred,data[7][idx:idx+1],data[5][idx:idx+1],data[3][idx:idx+1],data[4][idx:idx+1])
   validation_rows[n].append({**records['validation'][j],'metrics':dict(zip(s.METRIC_NAMES,map(float,vals)))})
  if j%16==15:print('PhysenNet validation',j+1,flush=True)
 tuning=[{'iterations':n,'macro':s.macro(rows),'score':s.score(s.macro(rows))} for n,rows in validation_rows.items()]
 chosen=max(tuning,key=lambda r:r['score'])['iterations'];s.dump(folder/'selection.json',{'selected_iterations':chosen,'candidates':tuning,'rule':'one global validation-selected iteration budget; no per-evaluation target stopping'})
 rows=[];preds=[];timings=[]
 for j,idx in enumerate(split['holdout']):
  cache=folder/'evaluation'/f'{j:03d}.pt'
  if cache.exists():obj=torch.load(cache,map_location='cpu',weights_only=True)
  else:
   b=[t[idx:idx+1].cuda() for t in data];pred,seconds=fit(b,[chosen]);obj={'prediction':pred[chosen],'seconds':seconds}
   cache.parent.mkdir(exist_ok=True);s.save(cache,obj)
  p=obj['prediction'];vals=s.compute_validation_metrics(p,data[7][idx:idx+1],data[5][idx:idx+1],data[3][idx:idx+1],data[4][idx:idx+1])
  rows.append({**records['holdout'][j],'metrics':dict(zip(s.METRIC_NAMES,map(float,vals)))});preds.append(p);timings.append(obj['seconds']*1000)
  if j%16==15:print('PhysenNet evaluation',j+1,flush=True)
 s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':torch.cat(preds)})
 s.dump(folder/'holdout_metrics.json',rows);s.dump(folder/'complete.json',{'status':'complete','macro':s.macro(rows),
  'config':{'iterations':chosen,'lr':.001,'initialization':'fresh random seed 101 per observation','training':'per-observation physical fitting, no supervised labels',
   'architecture':'PyTorch port of authors 32-512-channel U-Net; paired propagation and observed-amplitude adaptation'},
  'latency_ms':{'median':float(np.median(timings)),'p10':float(np.quantile(timings,.1)),'p90':float(np.quantile(timings,.9)),
    'scope':'per-observation optimization runtime, excludes model allocation'},
  'script_sha256':s.sha(__file__)})
if __name__=='__main__':main()
