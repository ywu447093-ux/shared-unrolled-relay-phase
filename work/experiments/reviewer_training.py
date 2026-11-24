"""Fixed-budget control separating extra training from object-class diversification."""
from pathlib import Path
import argparse,copy,json,time,math,random
from collections import Counter
import numpy as np,torch
import suite as s
import calibrated
import reviewer_validation as v
ROOT=s.ROOT;OUT=ROOT/'work/reviewer_training_v1';OUT.mkdir(exist_ok=True)
ARMS={'stage4_mixed':('stage4','mixed'),'stage4_coupled':('stage4','coupled'),'unet_mixed':('calibrated_unet','mixed'),'unet_coupled':('calibrated_unet','coupled')}

def prepare():
    if (OUT/'preparation_complete.json').exists():return
    rec=json.loads((s.OLD/'outputs/R1_controlled_experiments_v2/source_split.json').read_text(encoding='utf8'))
    cfg={'stage':'frozen before corrective training','code_sha256':s.sha(__file__),'time_unix':time.time(),'arms':ARMS,
      'purpose':'Separate additional optimization from mixed object-class training after original models failed independent-absorption stress testing.',
      'source_partitions':'Use only existing train source contents for optimization and existing validation contents for selection. Secondary sources and internal evaluation contents are excluded from generated training and validation.',
      'seeds':{'training':101,'train_generation':2026110500,'validation_generation':2026110600},
      'warmstarts':{a:s.sha(s.OUT/'learning'/a/'best.pt') for a in ['stage4','calibrated_unet']},
      'additional_epochs':40,'observations_per_epoch':1184,'batch_size':8,'peak_lr':1e-4,'optimizer':'AdamW, weight_decay 1e-5, gradient_clip 1, warmup3 then cosine to1%',
      'sampling':'Same source and realization IDs across arms. Mixed uses balanced coupled/independent-absorption/pure-phase observations; coupled replaces each sampled condition by coupled at the same source and realization. Same joint D4 transforms.',
      'objective':'Unchanged final-output phase/amplitude/physical objective.',
      'selection':'Same composite score, macro averaged over the three generated validation conditions and five categories. No secondary or internal evaluation targets used.',
      'scope':'Post hoc training-distribution experiment; seed101, unchanged architecture. Secondary cohort already inspected to motivate this experiment; no prospective validation claim.'}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==cfg
    else:s.dump(OUT/'protocol.json',cfg)
    for split,seedbase in [('train',2026110500),('validation',2026110600)]:
        path=OUT/(split+'_cache.pt')
        if path.exists():continue
        sources=sorted({r['content_id']:(r['source_qpm'],r['cell_type']) for r in rec[split]}.items())
        qs=[]
        for i,(cid,(p,c)) in enumerate(sources):
            crop=v.gen.content_aware_native_crop(v.gen._read_qpm(p),128,np.random.default_rng(seedbase+i))
            qs.append(v.gen.qpm_to_phase_fraction(crop))
        samples=[];rows=[]
        for i,(cid,(p,c)) in enumerate(sources):
            donor=next((i+j)%len(sources) for j in range(1,len(sources)) if sources[(i+j)%len(sources)][1][1]!=c)
            for rep in range(2):
                for ci,condition in enumerate(v.CONDITIONS[1:]):
                    # Distinct seed ranges from secondary object-class tests.
                    seed_index=i+(10000 if split=='train' else 20000)
                    values,meta=v.pack_sample(qs[i],qs[donor],seed_index,rep,condition)
                    samples.append(values);rows.append({'name':f'{split}_{i:04d}_{rep}_{ci}','source_qpm':p,'content_id':cid,'cell_type':c,'condition':condition,'condition_index':ci,'source_index':i,'simulation_index':rep,**meta})
            if (i+1)%50==0:print('PREP',split,i+1,'/',len(sources),flush=True)
        data=[torch.stack([x[k] for x in samples]) for k in range(8)]
        assert len(data[0])==len(sources)*6
        s.save(path,{'data':data,'records':rows,'sources':sources})
        print('PREPARED',split,len(sources),'contents',len(rows),'observations',flush=True)
    train=torch.load(OUT/'train_cache.pt',weights_only=False);val=torch.load(OUT/'validation_cache.pt',weights_only=False)
    trainids={r['content_id'] for r in train['records']};valids={r['content_id'] for r in val['records']}
    secondary={r['content_id'] for r in json.loads((v.OUT/'secondary_source_records.json').read_text())}
    assert not trainids&valids and not trainids&secondary and not valids&secondary
    assert len(trainids)==294 and len(valids)==64
    s.dump(OUT/'preparation_complete.json',{'status':'complete','train_contents':294,'validation_contents':64,'all_partitions_disjoint':True})

def train_arm(arm,train,val):
    folder=OUT/arm;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    base,mode=ARMS[arm];s.seed(101);model=s.model_for(base).cuda()
    initial=torch.load(s.OUT/'learning'/base/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(initial['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:min((ep+1)/3,1)*(.01+.99*.5*(1+math.cos(math.pi*max(ep-3,0)/37))))
    physics=s.DeterministicPhysicalLoss().cuda();history=[];best=-float('inf');start=0;elapsed=0
    if (folder/'resume.pt').exists():
        ck=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        history=ck['history'];start=ck['epoch'];best=ck['best'];elapsed=ck['elapsed']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    data=train['data'];t0=time.time()
    for ep in range(start,40):
        rng=np.random.default_rng(101+500000+ep);parents=rng.integers(294,size=1184);reps=rng.integers(2,size=1184)
        conditions=np.arange(1184)%3;rng.shuffle(conditions)
        if mode=='coupled':conditions=np.zeros_like(conditions)
        order=parents*6+reps*3+conditions;model.train();losses=[]
        for off in range(0,1184,8):
            b=[t[order[off:off+8]].cuda() for t in data];k=int(rng.integers(4));flip=int(rng.integers(2))
            for ch in [0,3,4,5,6,7]:b[ch]=s.d4(b[ch],k,flip)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
            with torch.autocast('cuda',enabled=False):loss=s.objective(pred,b,physics,False)
            assert torch.isfinite(loss),(arm,ep)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        sched.step();macro,rows,_=s.evaluate(model,val['data'],list(range(len(val['records']))),val['records'])
        value=s.score(macro);bycondition={c:s.macro([r for r in rows if r['condition']==c]) for c in v.CONDITIONS[1:]}
        row={'epoch':ep+1,'loss':float(np.mean(losses)),'selection_score':value,'validation_macro':macro,'validation_conditions':bycondition,'elapsed_s':elapsed+time.time()-t0}
        history.append(row);s.dump(folder/'history.json',history)
        if value>best:
            best=value;s.save(folder/'best.pt',{'model':model.state_dict(),'epoch':ep+1,'score':value,'arm':arm,'protocol_sha256':s.sha(OUT/'protocol.json')})
        s.save(folder/'resume.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history,'epoch':ep+1,'best':best,'elapsed':row['elapsed_s'],'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        print(json.dumps({'arm':arm,'epoch':ep+1,'validation_ssim':{c:bycondition[c]['phase_foreground_ssim'] for c in bycondition},'elapsed_s':row['elapsed_s']}),flush=True)
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    s.dump(folder/'complete.json',{'status':'complete','selected_epoch':ck['epoch'],'validation_score':ck['score'],'checkpoint_sha256':s.sha(folder/'best.pt'),'elapsed_s':history[-1]['elapsed_s']})
    del model,opt,physics;torch.cuda.empty_cache()

def evaluate_arm(arm):
    base,_=ARMS[arm];model=s.model_for(base).cuda().eval();ck=torch.load(OUT/arm/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    for condition in v.CONDITIONS:
        folder=OUT/'evaluation'/condition/arm;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():continue
        cached=torch.load(v.OUT/(condition+'_cache.pt'),weights_only=False)
        rows=[];preds=[]
        with torch.inference_mode():
            for off in range(0,len(cached['records']),4):
                b=[t[off:off+4].cuda() for t in cached['data']]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
                assert torch.isfinite(pred).all()
                for j,r in enumerate(cached['records'][off:off+4]):
                    vv=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1]);metrics=dict(zip(s.METRIC_NAMES,map(float,vv)));metrics.update(v.phase_metrics(pred[j:j+1,1:2],b[7][j:j+1,1:2]))
                    rows.append({**r,'metrics':metrics})
                preds.append(pred.cpu())
        macro={k:float(np.mean([np.mean([r['metrics'][k] for r in rows if r['cell_type']==c]) for c in sorted({r['cell_type'] for r in rows})])) for k in rows[0]['metrics']}
        s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':torch.cat(preds)})
        s.dump(folder/'per_observation_metrics.json',rows);s.dump(folder/'complete.json',{'status':'complete','macro':macro,'source_contents':105,'observations':210,'checkpoint_sha256':s.sha(OUT/arm/'best.pt')})
        print('EVALUATED',condition,arm,macro['phase_foreground_rmse_rad'],flush=True)
    del model;torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;calibrated.register();prepare()
    if args.prepare_only:return
    train=torch.load(OUT/'train_cache.pt',weights_only=False);val=torch.load(OUT/'validation_cache.pt',weights_only=False)
    for arm in ARMS:train_arm(arm,train,val)
    del train,val
    for arm in ARMS:evaluate_arm(arm)
    s.dump(OUT/'all_complete.json',{'status':'complete','arms':list(ARMS),'additional_epochs_each':40})
if __name__=='__main__':main()
