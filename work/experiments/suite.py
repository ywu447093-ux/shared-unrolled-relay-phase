"""Single-seed controlled exploration for the revised relay-screen manuscript.

All new supervised runs start from random initialization. Archived results are
read-only and are never substituted for the matched new reference.
"""
from pathlib import Path
import argparse, copy, csv, hashlib, importlib.util, json, math, os, random, sys, time
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=Path(r'C:/Users/Lenovo/Documents/Codex/2026-08-30/ncpp-ncpp-ncpp-annotated-quantitative-phase')
CODE=ROOT/'work/code/NCPP_V12'
sys.path[:0]=[str(CODE/'work'),str(CODE/'outputs/ncpp_physical_screen_v11'),str(HERE/'vendor')]
from compact_ncpp_v11_model import model_for as conv_model
from compact_ncpp_v12_model import GlobalMixingBlock
from benchmark_ncpp_runtime import make_propagators
from real_model import DeterministicPhysicalLoss,_ssim_map,_camera_modulus_gain
from train_real_cells import compute_validation_metrics, METRIC_NAMES

OUT=ROOT/'work/experiment_results_v1'
ARMS=['stage4','stage1','stage2','stage3','stage6','stage8','no_global','no_attention','heads1','heads8',
      'private_heads','private_decoders','unshared','direct_unet','direct_rescnn','sisprnet_adapted','amplitude_only']
LABELS={'stage4':'Four stages, shared, four heads','stage1':'One stage','stage2':'Two stages','stage3':'Three stages',
 'stage6':'Six stages','stage8':'Eight stages','no_global':'Convolutional bottleneck','no_attention':'Mixer FFN without attention',
 'heads1':'One attention head','heads8':'Eight attention heads','private_heads':'Stage-specific output heads',
 'private_decoders':'Shared encoder and mixer, private decoders','unshared':'Independent stage networks',
 'direct_unet':'Direct encoder-decoder','direct_rescnn':'Direct residual CNN','sisprnet_adapted':'SiSPRNet adapted to paired observations',
 'amplitude_only':'Amplitude-only encoder-decoder'}

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def d4(x,k,flip):
 for _ in range(k):x=torch.roll(x.transpose(-2,-1).flip(-2),1,-2)
 if flip:x=torch.roll(x.flip(-1),1,-1)
 return x.contiguous()

def fast_attention(self,x):
 batch,channels,height,width=x.shape;tokens=height*width
 qkv=self.qkv(self.normalized(self.norm1,x)).reshape(batch,3,self.heads,channels//self.heads,tokens)
 q,k,v=[t.transpose(-2,-1).contiguous() for t in qkv.unbind(1)]
 mixed=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
 return mixed.transpose(-2,-1).reshape(batch,channels,height,width)

class FeedForwardOnly(nn.Module):
 def __init__(self,block):super().__init__();self.norm2=copy.deepcopy(block.norm2);self.ffn=copy.deepcopy(block.ffn)
 def forward(self,x):return x+.1*self.ffn(GlobalMixingBlock.normalized(self.norm2,x))

class StagePrior(nn.Module):
 def __init__(self,prior,stages,mode):
  super().__init__();self.mode=mode
  if mode=='unshared':self.networks=nn.ModuleList([copy.deepcopy(prior) for _ in range(stages)]);return
  self.shared=prior
  if mode=='private_heads':
   self.heads=nn.ModuleList([copy.deepcopy(prior.head) for _ in range(stages)]);self.shared.head=nn.Identity()
  if mode=='private_decoders':
   self.decoders=nn.ModuleList([nn.ModuleDict({k:copy.deepcopy(getattr(prior,k)) for k in ['up1','up2','head']}) for _ in range(stages)])
   self.shared.up1=self.shared.up2=self.shared.head=nn.Identity()
 def forward(self,x,progress,index):
  if self.mode=='unshared':return self.networks[index](x,progress)
  if self.mode=='private_heads':return self.heads[index](self.shared(x,progress))
  if self.mode=='private_decoders':
   p=self.shared;dd=self.decoders[index]
   stem=p.stem[0](x)+progress*p.stage_bias[None,:,None,None]
   f0=p.stem[2](p.stem[1](stem));f1=p.down1(F.avg_pool2d(f0,2));f2=p.down2(F.avg_pool2d(f1,2))
   u1=dd['up1'](torch.cat([F.interpolate(f2,size=f1.shape[-2:],mode='bilinear',align_corners=False),f1],1))
   u2=dd['up2'](torch.cat([F.interpolate(u1,size=f0.shape[-2:],mode='bilinear',align_corners=False),f0],1))
   return dd['head'](u2)
  return self.shared(x,progress)

class Reconstructor(nn.Module):
 def __init__(self,arm):
  super().__init__();self.stages=int(arm[5:]) if arm.startswith('stage') else 4
  base=conv_model('dilate_wide48');prior=base.prior
  if arm!='no_global':
   block=GlobalMixingBlock(heads=1 if arm=='heads1' else 8 if arm=='heads8' else 4)
   block.attention_message=fast_attention.__get__(block,GlobalMixingBlock)
   prior.down2[3]=FeedForwardOnly(block) if arm=='no_attention' else block
  self.prior=StagePrior(prior,self.stages,arm)
  self.dc_logits=nn.Parameter(torch.full((self.stages,),math.log(.15/.85)))
 def forward(self,b):
  s,pn,pp,a,flat,sup,sat=b[:7];a=a.clamp(0,1.25);fl=flat.clamp_min(1e-4)
  phase=torch.zeros_like(a);amp=a;prop,inv=make_propagators(pp,s.shape[-1]);gain=_camera_modulus_gain(pp)
  mu=(s*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/gain
  qq=torch.quantile(s.flatten(2),torch.tensor([.08,.30],device=s.device),dim=2)
  low=qq[0,:,:,None,None];nu=(qq[1]-qq[0]).clamp_min(.005)[:,:,None,None]
  signal=(s-low).clamp_min(0);weight=.25+.75*signal/(signal+nu+1e-8)
  absorption=((fl-a)/fl).clamp(-.1,.15)*20;maps=pn[:,:3,None,None].expand(-1,-1,*a.shape[-2:])
  for k in range(self.stages):
   u=torch.polar(amp,phase);v=prop(u);mod=v.abs();target=torch.where((sat>.5)&(mod>mu),mod,mu)
   obj=inv((mod+weight*(target-mod))*v/(mod+1e-8));residual=(obj-u)/fl
   features=torch.cat([phase/math.pi,residual.real.clamp(-2,2),residual.imag.clamp(-2,2),absorption,sup.clamp(0,1),flat,maps],1)
   delta=self.prior(features,k/max(self.stages-1,1),k).float()
   phase=(phase+torch.sigmoid(self.dc_logits[k])*torch.sin(torch.angle(obj*torch.conj(u)))+.5*math.pi*torch.tanh(delta[:,1:2])).clamp(0,math.pi)
   amp=(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25)
  return torch.cat([amp,phase],1)

class Direct(nn.Module):
 def __init__(self,arm):
  super().__init__();self.arm=arm
  if arm=='sisprnet_adapted':
   from model_SiPRNet_original import MyNet
   self.backbone=MyNet();self.fusion=nn.Sequential(nn.Conv2d(9,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,2,3,padding=1))
  elif arm=='direct_rescnn':
   from compact_ncpp_model import Residual
   self.net=nn.Sequential(nn.Conv2d(9,64,3,padding=1),nn.SiLU(),*[Residual(64) for _ in range(8)],nn.Conv2d(64,2,3,padding=1))
  else:
   self.net=conv_model('dilate_wide48').prior
 def forward(self,b):
  s,pn,pp,a,flat,sup,sat=b[:7];maps=pn[:,:3,None,None].expand(-1,-1,*s.shape[-2:])
  if self.arm=='sisprnet_adapted':
   z=self.backbone(s);delta=self.fusion(torch.cat([z,a,flat,sup,sat,maps],1)).float()
  else:
   if self.arm=='amplitude_only':
    z=torch.zeros_like(s);x=torch.cat([a,flat,sup,((flat-a)/flat.clamp_min(1e-4)).clamp(-.1,.15)*20,maps,z,z],1)
   else:x=torch.cat([s,torch.log1p(100*s)/math.log(101),a,flat,sup,sat,maps],1)
   delta=(self.net(x,0.) if self.arm!='direct_rescnn' else self.net(x)).float()
  return torch.cat([(a.clamp(0,1.25)+.03*torch.tanh(delta[:,:1])).clamp(0,1.25),math.pi*torch.sigmoid(delta[:,1:2])],1)

def model_for(arm):return Direct(arm) if arm.startswith('direct') or arm in ['sisprnet_adapted','amplitude_only'] else Reconstructor(arm)

def load_data(aug=True):
 data=torch.load(OLD/'outputs/R1_controlled_experiments_v2/training_cache.pt',map_location='cpu',weights_only=True)
 assert len(data)==8 and len(data[0])==848
 names=sorted(x.stem for x in (OLD/'work/real_cell_dataset_v11/sps').glob('train_*.npy'));lookup={n:i for i,n in enumerate(names)}
 records=json.loads((OLD/'outputs/R1_controlled_experiments_v2/source_split.json').read_text(encoding='utf8'))
 split={k:[lookup[r['name']] for r in rows] for k,rows in records.items()}
 assert [len(split[k]) for k in ['train','validation','holdout']]==[592,128,128]
 parent_groups=None
 if aug:
  extra=torch.load(OLD/'outputs/compact_ncpp_v6/augmented_cache.pt',map_location='cpu',weights_only=True)
  data=[torch.cat([a,b],0) for a,b in zip(data,extra)]
  manifest=json.loads((OLD/'outputs/compact_ncpp_v6/augmentation_manifest.json').read_text(encoding='utf8'))
  groups=defaultdict(list)
  for row in manifest:groups[row['content_id']].append(row['index']+848)
  parents={r['content_id'] for r in records['train']};assert set(groups)==parents
  assert not parents&{r['content_id'] for r in records['holdout']+records['validation']}
  parent_groups=[np.array(groups[k]) for k in sorted(groups)]
 return data,split,records,parent_groups

def objective(pred,b,physics,amplitude_only=False):
 target=b[7];truth=target[:,1:2];fg=(truth>torch.maximum(.05*truth.amax((-2,-1),keepdim=True),truth.new_tensor(.03))).float()
 w=1+4*fg;error=(pred[:,1:2]-truth)/math.pi
 fidelity=((torch.sqrt(error.square()+1e-6)*w).sum((1,2,3))/w.sum((1,2,3))).mean()
 sm=_ssim_map(pred[:,1:2],truth,data_range=math.pi)
 structure=(1-(sm*fg).sum((1,2,3))/(fg.sum((1,2,3))+1e-8)).mean()
 gradient=0.
 for dim in [-1,-2]:
  dx=torch.diff(error,dim=dim);sl0=[slice(None)]*4;sl1=sl0.copy();sl0[dim]=slice(None,-1);sl1[dim]=slice(1,None)
  wx=1+4*torch.maximum(fg[tuple(sl0)],fg[tuple(sl1)])
  gradient=gradient+((torch.sqrt(dx.square()+1e-6)*wx).sum((1,2,3))/wx.sum((1,2,3))).mean()
 amp=20*(pred[:,:1]-target[:,:1]).abs().mean()+1-_ssim_map(pred[:,:1],target[:,:1],data_range=1.).mean()
 # A strict amplitude-only baseline never uses SPS even through its objective.
 physical=0. if amplitude_only else .005*physics(pred[:,:1],pred[:,1:2],b[0],b[2],saturation_mask=b[6])
 return 8*(2*fidelity+2*structure+gradient)+amp+physical

def macro(rows):
 cats=sorted({r['cell_type'] for r in rows})
 return {k:float(np.mean([np.mean([r['metrics'][k] for r in rows if r['cell_type']==c]) for c in cats])) for k in METRIC_NAMES}
def score(m):return m['phase_foreground_ssim']+.2*m['phase_gradient_correlation']-2*max(0,-m['amp_ssim_gain']-.003)

@torch.inference_mode()
def evaluate(model,data,ids,records,keep=False,amp=True):
 rows=[];preds=[];model.eval()
 for off in range(0,len(ids),4):
  b=[t[ids[off:off+4]].cuda() for t in data]
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):pred=model(b).float()
  assert torch.isfinite(pred).all()
  for j,row in enumerate(records[off:off+4]):
   values=compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1])
   rows.append({**row,'metrics':dict(zip(METRIC_NAMES,map(float,values)))})
  if keep:preds.append(pred.cpu())
 return macro(rows),rows,torch.cat(preds) if keep else None

def train(arm,args,data,split,records,groups):
 folder=OUT/'learning'/arm;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 seed(args.seed);model=model_for(arm).cuda();opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:min((ep+1)/3,1)*(.01+.99*.5*(1+math.cos(math.pi*max(ep-3,0)/max(args.epochs-3,1)))))
 cfg={'arm':arm,'label':LABELS[arm],'seed':args.seed,'epochs':args.epochs,'lr':args.lr,'effective_batch':8,'microbatch':args.microbatch,
  'initialization':'random; no pretrained weights','sampling':'592 original plus 592 parent-uniform frozen augmented observations per epoch',
  'precision':'BF16 neural layers; FP32 propagation, final states, losses and metrics; automatic SDPA backend',
  'loss':'V12 final-output objective; physical term removed for strict amplitude-only control','parameters':sum(p.numel() for p in model.parameters()),
  'script_sha256':sha(__file__),'vendor_model_sha256':sha(HERE/'vendor/model_SiPRNet_original.py') if arm=='sisprnet_adapted' else None,
  'source_split_sha256':sha(OLD/'outputs/R1_controlled_experiments_v2/source_split.json'),
  'selection':'validation-only category-macro composite score; holdout previously reused during historical method development',
  'budget_interpretation':'equal example and optimizer-step budgets, not equal compute or verified convergence'}
 if (folder/'config.json').exists():assert json.loads((folder/'config.json').read_text())==cfg
 else:dump(folder/'config.json',cfg)
 history=[];start=0;best=-float('inf');elapsed=0
 if (folder/'resume.pt').exists():
  ck=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
  history=ck['history'];start=ck['epoch'];best=ck['best'];elapsed=ck['elapsed']
  random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 physics=DeterministicPhysicalLoss().cuda();begin=time.perf_counter()
 for ep in range(start,args.epochs):
  model.train();rng=np.random.default_rng(args.seed+100000+ep)
  aug=np.array([rng.choice(groups[int(parent)]) for parent in rng.integers(len(groups),size=592)])
  order=np.concatenate([np.array(split['train']),aug]);rng.shuffle(order);losses=[]
  for off in range(0,len(order),8):
   group=order[off:off+8];opt.zero_grad(set_to_none=True)
   for j in range(0,len(group),args.microbatch):
    ix=group[j:j+args.microbatch];b=[t[ix].cuda() for t in data];k=int(rng.integers(4));flip=int(rng.integers(2))
    for channel in [0,3,4,5,6,7]:b[channel]=d4(b[channel],k,flip)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
    with torch.autocast('cuda',enabled=False):loss=objective(pred,b,physics,arm=='amplitude_only')
    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {arm} epoch {ep+1}')
    (loss*len(ix)/len(group)).backward();losses.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  sched.step();val,_,_=evaluate(model,data,split['validation'],records['validation']);value=score(val)
  row={'epoch':ep+1,'loss':float(np.mean(losses)),'validation_macro':val,'selection_score':value,'elapsed_s':elapsed+time.perf_counter()-begin}
  history.append(row);dump(folder/'history.json',history)
  if value>best:
   best=value;save(folder/'best.pt',{'model':model.state_dict(),'epoch':ep+1,'score':value,'config':cfg})
  save(folder/'resume.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,
   'epoch':ep+1,'best':best,'elapsed':row['elapsed_s'],'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
  print(json.dumps({'arm':arm,'epoch':ep+1,'fg':val['phase_foreground_ssim'],'gradient':val['phase_gradient_correlation'],'elapsed_s':row['elapsed_s']}),flush=True)
 ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
 mm,rows,preds=evaluate(model,data,split['holdout'],records['holdout'],True)
 save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':preds})
 dump(folder/'holdout_metrics.json',rows)
 dump(folder/'complete.json',{'status':'complete','best_epoch':ck['epoch'],'macro':mm,'checkpoint_sha256':sha(folder/'best.pt'),'elapsed_s':history[-1]['elapsed_s']})
 del model,opt,physics;torch.cuda.empty_cache()

def smoke(args,data,split,records):
 report={};b=[t[split['train'][:2]].cuda() for t in data];physics=DeterministicPhysicalLoss().cuda()
 for arm in ARMS:
  seed(args.seed);m=model_for(arm).cuda().train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(b).float()
  loss=objective(pred,b,physics,arm=='amplitude_only');loss.backward();torch.cuda.synchronize()
  assert pred.shape==b[7].shape and torch.isfinite(pred).all() and torch.isfinite(loss)
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  report[arm]={'parameters':sum(p.numel() for p in m.parameters()),'two_image_forward_backward_s':time.perf_counter()-start,
   'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'loss':float(loss.detach())}
  print(arm,report[arm],flush=True);del m,loss,pred;torch.cuda.empty_cache()
 # Decoupling starts from the same function, with copied rather than randomized stages.
 reference=None
 for arm in ['stage4','private_heads','private_decoders','unshared']:
  seed(args.seed);m=model_for(arm).cuda().eval()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pp=m(b).float()
  if reference is None:reference=pp.clone()
  else:assert float((pp-reference).abs().max())<1e-5,(arm,float((pp-reference).abs().max()))
  del m;torch.cuda.empty_cache()
 # The amplitude-only forward map must be invariant to all SPS inputs.
 seed(args.seed);m=model_for('amplitude_only').cuda().eval();bb=[t.clone() for t in b]
 bb[0]=torch.rand_like(bb[0]);bb[2]=torch.rand_like(bb[2]);bb[6]=1-bb[6]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  assert torch.equal(m(b),m(bb))
 dump(OUT/'smoke.json',report)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--seed',type=int,default=101)
 ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--microbatch',type=int,default=8);ap.add_argument('--arms',nargs='+',default=ARMS,choices=ARMS);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 OUT.mkdir(parents=True,exist_ok=True);data,split,records,groups=load_data()
 if args.smoke:smoke(args,data,split,records)
 else:
  for arm in args.arms:train(arm,args,data,split,records,groups)
if __name__=='__main__':main()
