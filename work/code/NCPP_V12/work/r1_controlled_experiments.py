"""Fresh, source-disjoint R1 comparisons. Never initialize from archived weights.

The historical validation set is excluded. The new holdout is isolated from all
training and checkpoint selection in this suite, but its source images were
available to the historical project; it is not a newly collected external cohort.
"""
from pathlib import Path
import argparse, copy, csv, hashlib, json, math, random, sys, time
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'outputs'/'ncpp_physical_screen_v11'
OUT = ROOT/'outputs'/'R1_controlled_experiments_v2'
sys.path.insert(0, str(CODE))
from real_model import (PhysicsUnrolledGenerator, FrequencyPreservingPrior,
    ConvAct, ResidualBlock, DeterministicPhysicalLoss, _ssim_map,
    adaptive_forward_prop_vec)
from real_GAN_dataset import RelaySPSDataset
from train_real_cells import mixed_loss_unrolled, compute_validation_metrics, METRIC_NAMES

ARMS = ['full6','rescnn','unet','stage1','stage3','stage9',
        'no_iterative_dc','no_detail','no_context','unshared','unbalanced']

def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def d4(x, k, flip):
    # Origin-centred permutations on even FFT grids, not half-pixel rotations.
    for _ in range(k): x = torch.roll(x.transpose(-2,-1).flip(-2),1,-2)
    if flip: x = torch.roll(x.flip(-1),1,-1)
    return x.contiguous()

class BranchPrior(FrequencyPreservingPrior):
    def __init__(self, original, mode):
        nn.Module.__init__(self); self.mode=mode
        self.stem=copy.deepcopy(original.stem); self.fuse=copy.deepcopy(original.fuse)
        if mode!='no_detail': self.detail=copy.deepcopy(original.detail)
        if mode!='no_context':
            for key in ['down1','down2','up1','up2']: setattr(self,key,copy.deepcopy(getattr(original,key)))
    def forward(self,x):
        h1,h2=self._high_pass_features(x); f0=self.stem(torch.cat([x,h1,h2],1))
        detail=self.detail(f0) if self.mode!='no_detail' else torch.zeros_like(f0)
        if self.mode=='no_context': return self.fuse(detail)
        f1=self.down1(f0);f2=self.down2(f1)
        u1=self.up1(torch.cat([F.interpolate(f2,size=f1.shape[-2:],mode='nearest'),f1],1))
        u2=self.up2(torch.cat([F.interpolate(u1,size=f0.shape[-2:],mode='nearest'),f0],1))
        return self.fuse(u2+detail)

class IdentityDC(nn.Module):
    def forward(self,amp,phase,*args,**kwargs):
        u=torch.polar(amp,phase); return u.real,u.imag

class ImageBaseline(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind=kind
        if kind=='rescnn':
            self.net=nn.Sequential(ConvAct(10,64),*[ResidualBlock(64,dilation=1 if i%2==0 else 2) for i in range(8)],nn.Conv2d(64,2,3,padding=1))
        else:
            self.stem=ConvAct(10,48)
            self.down1=nn.Sequential(ConvAct(48,96),ResidualBlock(96))
            self.down2=nn.Sequential(ConvAct(96,192),ResidualBlock(192),ResidualBlock(192))
            self.up1=ConvAct(288,96);self.up2=ConvAct(144,48)
            self.head=nn.Conv2d(48,2,3,padding=1)
    def forward(self,s,pn,pp,amplitude_measurement,amplitude_flat_field,support_mask,saturation_mask,**kw):
        a=amplitude_measurement
        x=torch.cat([s,torch.log1p(100*s.clamp(0,1))/math.log(101),
            s-F.avg_pool2d(s,5,1,2),a,amplitude_flat_field,support_mask,saturation_mask,
            pn[:,:3,None,None].expand(-1,-1,*s.shape[-2:])],1)
        if self.kind=='rescnn': delta=self.net(x)
        else:
            f0=self.stem(x);f1=self.down1(F.avg_pool2d(f0,2));f2=self.down2(F.avg_pool2d(f1,2))
            u1=self.up1(torch.cat([F.interpolate(f2,size=f1.shape[-2:],mode='nearest'),f1],1))
            u2=self.up2(torch.cat([F.interpolate(u1,size=f0.shape[-2:],mode='nearest'),f0],1));delta=self.head(u2)
        return torch.cat([(a+.05*delta[:,:1]).clamp(0,1.25),(.5*delta[:,1:2]).clamp(0,math.pi)],1)

def model_for(arm):
    if arm in ['rescnn','unet']: return ImageBaseline(arm)
    stages={'stage1':1,'stage3':3,'stage9':9}.get(arm,6)
    m=PhysicsUnrolledGenerator(num_stages=stages,features=48,img_size=128,shared_prior=arm!='unshared')
    if arm=='no_iterative_dc': m.dc_blocks=nn.ModuleList([IdentityDC() for _ in range(stages)])
    if arm in ['no_detail','no_context']:
        m.prior=BranchPrior(m.prior,arm);m.output_refiner=BranchPrior(m.output_refiner,arm)
    return m

def forward(m,b):
    s,pn,pp,a,flat,sup,sat,gt=b
    result=m(s,pn,pp,amplitude_measurement=a,amplitude_flat_field=flat,support_mask=sup,saturation_mask=sat)
    return result[-1] if isinstance(result,list) else result

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=list(csv.DictReader((ROOT/'work'/'real_cell_dataset_v11'/'manifest.csv').open(encoding='utf-8-sig')))
    ds=RelaySPSDataset(str(ROOT/'work'/'real_cell_dataset_v11'),'train',augment=False)
    names=[Path(p).stem for p in ds.file_list];lookup={r['name']:r for r in manifest}
    groups={};identity={};identity_labels={}
    for name in names:
        r=lookup[name];source=r['source_qpm']
        if source not in identity:
            raw=cv2.imread(source,cv2.IMREAD_UNCHANGED)
            if raw is None:raise FileNotFoundError(source)
            identity[source]=hashlib.sha256(str((raw.shape,str(raw.dtype))).encode()+raw.tobytes()).hexdigest()
        content_id=identity[source];identity_labels.setdefault(content_id,set()).add(r['cell_type'])
        groups.setdefault(r['cell_type'],set()).add(content_id)
    assert all(len(labels)==1 for labels in identity_labels.values()), 'Identical image has inconsistent category labels'
    sets={'train':set(),'validation':set(),'holdout':set()};rng=np.random.default_rng(20261019)
    for cls,values in sorted(groups.items()):
        values=sorted(values);rng.shuffle(values);n=max(1,round(.15*len(values)))
        sets['holdout'].update(values[:n]);sets['validation'].update(values[n:2*n]);sets['train'].update(values[2*n:])
    split={k:[i for i,n in enumerate(names) if identity[lookup[n]['source_qpm']] in v] for k,v in sets.items()}
    assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','holdout'),('validation','holdout')])
    records={k:[{**lookup[names[i]],'content_id':identity[lookup[names[i]]['source_qpm']]} for i in ids] for k,ids in split.items()}
    path=OUT/'source_split.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf8'))==records,'Refuse changed split'
    else:dump(path,records)
    cache=OUT/'training_cache.pt'
    if cache.exists(): data=torch.load(cache,weights_only=True)
    else:
        all_samples=[ds[i] for i in range(len(ds))]
        data=[torch.stack([r[k] for r in all_samples]) for k in range(8)];torch.save(data,cache)
    return data,split,records

def evaluate(m,data,ids,batch=8,save=False):
    m.eval();metrics=[];preds=[]
    with torch.inference_mode():
        for off in range(0,len(ids),batch):
            ix=ids[off:off+batch];b=[t[ix].cuda() for t in data];p=forward(m,b)
            for j in range(len(ix)):metrics.append(compute_validation_metrics(p[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1]))
            if save:preds.append(p.cpu())
    return np.asarray(metrics),torch.cat(preds) if save else None

def run_one(arm,s,args,data,split,records):
    dest=OUT/f'{arm}_seed{s}';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    seed(s);m=model_for(arm).cuda();opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=1e-6)
    config={'arm':arm,'seed':s,'epochs':args.epochs,'microbatch_size':args.batch,'effective_batch_size':args.effective_batch,'lr':args.lr,
        'weight_decay':1e-5,'optimizer':'AdamW','scheduler':'cosine minimum 1e-6',
        'initialization':'random; no historical weights','selection':'macro-category validation foreground SSIM + 0.2 gradient correlation',
        'training_loss':'existing mixed_loss_unrolled on final output only; physical term weight 1; detail_weight 1',
        'augmentation':'joint FFT-origin-centred D4 for all image channels; none for evaluation',
        'sampling':'inverse category count with replacement' if arm!='unbalanced' else 'uniform with replacement',
        'parameter_count':sum(p.numel() for p in m.parameters()),
        'source_split_sha256':sha(OUT/'source_split.json'),
        'suite_sha256':sha(__file__), 'model_source_sha256':sha(CODE/'real_model.py'),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    config_path=dest/'config.json'
    if config_path.exists():assert json.loads(config_path.read_text(encoding='utf8'))==config,'Configuration changed; use new output directory'
    else:dump(config_path,config)
    hist=[];best=-float('inf');start_epoch=0;elapsed=0.
    resume=dest/'resume.pt'
    if resume.exists():
        ck=torch.load(resume,weights_only=False,map_location='cpu');m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start_epoch=ck['epoch'];hist=ck['history'];best=ck['best'];elapsed=ck['elapsed_s']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    classes=[r['cell_type'] for r in records['train']];counts={c:classes.count(c) for c in set(classes)}
    weights=np.array([1/counts[c] for c in classes]) if arm!='unbalanced' else np.ones(len(classes));weights/=weights.sum()
    physics=DeterministicPhysicalLoss().cuda();start=time.perf_counter()
    for ep in range(start_epoch,args.epochs):
        m.train();order=np.random.choice(split['train'],len(split['train']),replace=True,p=weights);losses=[]
        for off in range(0,len(order),args.effective_batch):
            group=order[off:off+args.effective_batch];opt.zero_grad(set_to_none=True)
            for micro in range(0,len(group),args.batch):
                ix=group[micro:micro+args.batch];b=[t[ix].cuda() for t in data]
                k=random.randrange(4);flip=random.randrange(2)
                for j in [0,3,4,5,6,7]:b[j]=d4(b[j],k,flip)
                pred=forward(m,b)
                phys=physics(pred[:,:1],pred[:,1:2],b[0],b[2],saturation_mask=b[6])
                loss,_=mixed_loss_unrolled(pred,b[7],b[3],b[4],b[5],phys,detail_weight=1.)
                if not torch.isfinite(loss):raise FloatingPointError(f'{arm} epoch {ep+1}: nonfinite loss')
                (loss*(len(ix)/len(group))).backward();losses.append(loss.item())
            nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        sched.step();val,_=evaluate(m,data,split['validation']);cats=[r['cell_type'] for r in records['validation']]
        macro=np.mean([val[np.array(cats)==c].mean(0) for c in sorted(set(cats))],0);score=float(macro[6]+.2*macro[7])
        row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'validation_macro':dict(zip(METRIC_NAMES,map(float,macro))),'selection_score':score,'elapsed_s':elapsed+time.perf_counter()-start}
        hist.append(row);dump(dest/'history.json',hist)
        if score>best:
            best=score;torch.save({'model':m.state_dict(),'epoch':ep+1,'score':best,'config':config},dest/'best.pt')
        torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ep+1,'history':hist,'best':best,'elapsed_s':row['elapsed_s'],
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},resume)
        print(json.dumps({'arm':arm,'seed':s,'epoch':ep+1,'loss':row['train_loss'],'macro_fg':float(macro[6]),'elapsed_s':row['elapsed_s']}),flush=True)
    bestck=torch.load(dest/'best.pt',weights_only=False);m.load_state_dict(bestck['model']);metrics,preds=evaluate(m,data,split['holdout'],save=True)
    torch.save({'names':[r['name'] for r in records['holdout']],'predictions':preds},dest/'holdout_predictions.pt')
    rows=[{**r,'metrics':dict(zip(METRIC_NAMES,map(float,v)))} for r,v in zip(records['holdout'],metrics)]
    dump(dest/'holdout_metrics.json',rows)
    b=[t[split['holdout'][:1]].cuda() for t in data];m.eval()
    with torch.inference_mode():
        for _ in range(10):forward(m,b)
        times=[]
        for _ in range(50):
            torch.cuda.synchronize();t=time.perf_counter();forward(m,b);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t))
    dump(dest/'complete.json',{'best_epoch':bestck['epoch'],'best_validation_score':bestck['score'],
        'holdout_pooled':dict(zip(METRIC_NAMES,map(float,metrics.mean(0)))),
        'latency_ms_mean':float(np.mean(times)),'latency_ms_sd':float(np.std(times,ddof=1)),
        'checkpoint_sha256':sha(dest/'best.pt'),'prediction_sha256':sha(dest/'holdout_predictions.pt'),
        'elapsed_training_s':elapsed+time.perf_counter()-start})
    del m,opt;torch.cuda.empty_cache()

def smoke(data,split):
    seed(20261019);b=[t[split['train'][:2]].cuda() for t in data];report={}
    for arm in ARMS:
        m=model_for(arm).cuda().train();p=forward(m,b);p.square().mean().backward()
        assert p.shape==b[7].shape and torch.isfinite(p).all()
        report[arm]={'parameters':sum(v.numel() for v in m.parameters()),'shape':list(p.shape)}
        del m;torch.cuda.empty_cache()
    # Check augmentation against the actual discrete forward model.
    u=torch.polar(b[7][:,:1],b[7][:,1:2]);pp=b[2];lam=pp[:,0,None,None,None];z=pp[:,1,None,None,None];dx=lam*z/pp[:,2,None,None,None]
    field,_=adaptive_forward_prop_vec(u,z,lam,dx)
    errors=[]
    for k in range(4):
        for f in range(2):
            aug,_=adaptive_forward_prop_vec(d4(u,k,f),z,lam,dx)
            errors.append(float((aug.abs()-d4(field.abs(),k,f)).norm()/field.abs().norm()))
    report['D4_forward_modulus_relative_errors']=errors
    assert max(errors)<1e-4,errors
    dump(OUT/'smoke_verification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--effective-batch',type=int,default=8)
    ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    ap.add_argument('--arms',nargs='+',default=ARMS,choices=ARMS);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    data,split,records=prepare()
    if args.smoke:smoke(data,split)
    else:
        for s in args.seeds:
            for arm in args.arms:run_one(arm,s,args,data,split,records)
