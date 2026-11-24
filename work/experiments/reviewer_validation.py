"""Frozen-checkpoint secondary cohort and physically resimulated object-class tests.

No fitting, checkpoint selection, or hyperparameter search is performed here.
"""
from pathlib import Path
import argparse,csv,json,hashlib,math,time,sys
from collections import Counter,defaultdict
import cv2,numpy as np,torch
import suite as s
import calibrated
from real_GAN_dataset import RelaySPSDataset
import prepare_real_dataset as gen
from amplitude_support import estimate_support_from_amplitude
import classical
ROOT=s.ROOT;OUT=ROOT/'work/reviewer_validation_v1';OUT.mkdir(exist_ok=True)
NEURAL=['stage3','stage4','no_global','private_decoders','unshared','calibrated_unet','calibrated_sisprnet','amplitude_only']
CONDITIONS=['secondary','fresh_coupled','independent_absorption','pure_phase']
CLASSICAL={'secondary':['GS','HIO','HIO_ER','NCPP'],'fresh_coupled':['HIO','NCPP'],'independent_absorption':['HIO','NCPP'],'pure_phase':['HIO','NCPP']}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_hash(path):
    a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if a is None:raise FileNotFoundError(path)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def pack_sample(q,donor,i,rep,condition):
    seed=2026110400+i*10+rep
    a,phase,flat=gen.generate_qpm_proxy(q,np.random.default_rng(seed))
    if condition=='independent_absorption':a=gen.generate_qpm_proxy(donor,np.random.default_rng(seed))[0]
    if condition=='pure_phase':a=flat.copy()
    arng=np.random.default_rng(seed+1000000);gain=float(arng.uniform(.92,1.02))
    am=gen.simulate_amplitude_measurement(a,arng,gain=gain)
    ab=gen.simulate_amplitude_measurement(flat,arng,gain=gain)
    support=estimate_support_from_amplitude(am,flat_field=ab)
    obs,sat,pp,diagnostics=gen.generate_real_sps_corrected(phase,a,np.random.default_rng(seed+2000000))
    norm=lambda v,l,h:float(np.clip(2*(v-l)/(h-l+1e-12)-1,-1,1))
    pn=np.array([norm(pp[0],525e-9,540e-9),norm(pp[1],.85,1.15),norm(np.log10(pp[2]),-.66,-.45)],dtype=np.float32)
    t=lambda x:torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).unsqueeze(0)
    values=[t(obs),torch.from_numpy(pn),torch.tensor(pp,dtype=torch.float32),t(am),t(ab),t(support.soft),t(sat),torch.from_numpy(np.stack([a,phase]).astype(np.float32))]
    return values,{'field_seed':seed,'support_source':support.source,'support_valid':bool(support.valid),'saturation_fraction':float(sat.mean()),**diagnostics}

def prepare():
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text())['code_sha256']==sha(__file__),'Refuse a changed evaluated protocol.'
        return
    manifest=list(csv.DictReader((s.OLD/'work/real_cell_dataset_v11/manifest.csv').open(encoding='utf-8-sig')))
    prior=json.loads((s.OLD/'outputs/R1_controlled_experiments_v2/source_split.json').read_text(encoding='utf8'))
    development={r['content_id'] for rows in prior.values() for r in rows}
    sources={r['source_qpm']:r['cell_type'] for r in manifest if r['split']=='test'}
    records=[];excluded=[];seen=set();hashes={p:source_hash(p) for p in sorted(sources)}
    for r in manifest:
        if r['split']!='test':continue
        cid=hashes[r['source_qpm']]
        if cid in development:excluded.append({**r,'content_id':cid,'reason':'identical raw image content in the development pool'});continue
        records.append({**r,'content_id':cid})
    assert len(records)==210 and len({r['content_id'] for r in records})==105
    ckpts={arm:{'sha256':sha(s.OUT/'learning'/arm/'best.pt'),'selected_epoch':json.loads((s.OUT/'learning'/arm/'complete.json').read_text())['best_epoch']} for arm in NEURAL}
    configs={m:json.loads((s.OUT/'classical'/m/'complete.json').read_text())['config'] for m in CLASSICAL['secondary']}
    protocol={'stage':'written before any secondary/stress model result','code_sha256':sha(__file__),'time_unix':time.time(),
      'secondary_original_observations':212,'excluded_observations':len(excluded),'secondary_observations':len(records),'secondary_contents':105,
      'category_contents':dict(Counter({r['content_id']:r['cell_type'] for r in records}.values())),
      'source_disjoint_from_all_current_development_contents':True,
      'historical_status':'Original pre-existing test partition, absent from the current training/validation cache. Historical method-development use is unverified; retrospective secondary evaluation, not prospective/external validation.',
      'neural_methods':NEURAL,'conditions':CONDITIONS,'classical_methods':CLASSICAL,'classical_configs':configs,'checkpoints':ckpts,
      'stress_seed_base':2026110400,'stress_replicates':2,'crop_candidates':16,
      'paired_object_conditions':'Each source/realization shares phase, illumination, absorption strength, smoothing and optical draw across fresh conditions. Independent absorption uses the next different-category secondary source in sorted order; pure phase sets amplitude equal to illumination. Each observation is regenerated with the original intensity detector model and amplitude-derived support.',
      'selection':'No fitting, candidate/seed/epoch selection or target-phase alignment; all retained source contents and all failed cases included.',
      'metrics':'Same reference foreground, per-image errors in rad, within-content optical mean, equal-category macro, 2000 stratified content bootstrap. No seed uncertainty inferred.'}
    s.dump(OUT/'protocol.json',protocol);s.dump(OUT/'excluded_duplicate_sources.json',excluded)
    s.dump(OUT/'secondary_source_records.json',records)
    ds=RelaySPSDataset(str(s.OLD/'work/real_cell_dataset_v11'),'test',augment=False)
    idx={Path(p).stem:i for i,p in enumerate(ds.file_list)}
    samples=[ds[idx[r['name']]] for r in records]
    data=[torch.stack([x[k] for x in samples]) for k in range(8)]
    s.save(OUT/'secondary_cache.pt',{'data':data,'records':records})
    unique=sorted({r['source_qpm']:r['cell_type'] for r in records}.items());qs=[]
    for i,(p,c) in enumerate(unique):
        crop=gen.content_aware_native_crop(gen._read_qpm(p),128,np.random.default_rng(20261104+i))
        qs.append(gen.qpm_to_phase_fraction(crop))
    donor=[next((i+j)%len(unique) for j in range(1,len(unique)) if unique[(i+j)%len(unique)][1]!=c) for i,(p,c) in enumerate(unique)]
    for condition in CONDITIONS[1:]:
        samples=[];rows=[]
        for i,(path,c) in enumerate(unique):
            for rep in range(2):
                sample,meta=pack_sample(qs[i],qs[donor[i]],i,rep,condition)
                samples.append(sample);rows.append({'name':f'{condition}_{i:04d}_{rep}','source_qpm':path,'content_id':hashes[path],'cell_type':c,'simulation_index':str(rep),'donor_qpm':unique[donor[i]][0],**meta})
        values=[torch.stack([x[k] for x in samples]) for k in range(8)]
        s.save(OUT/(condition+'_cache.pt'),{'data':values,'records':rows})
    # Same phase and geometry across all fresh object conditions; inputs are finite.
    aa=torch.load(OUT/'fresh_coupled_cache.pt',weights_only=False)
    for condition in CONDITIONS[2:]:
        bb=torch.load(OUT/(condition+'_cache.pt'),weights_only=False)
        assert torch.equal(aa['data'][7][:,1],bb['data'][7][:,1])
        assert torch.equal(aa['data'][2][:,:3],bb['data'][2][:,:3])
        if condition=='pure_phase':assert all(torch.isfinite(t).all() for t in bb['data'])
    s.dump(OUT/'preparation_complete.json',{'status':'complete','conditions':CONDITIONS,'observations_each':210,'seconds_note':'No model results inspected during preparation'})
    print('Prepared secondary cohort and three paired object classes, 210 observations each.',flush=True)

def phase_metrics(pred,truth):
    mask=truth>torch.maximum(.05*truth.amax((-2,-1),keepdim=True),truth.new_tensor(.03))
    values=(pred-truth)[mask]
    return {'phase_foreground_mae_rad':float(values.abs().mean()),'phase_foreground_rmse_rad':float(values.square().mean().sqrt()),'phase_foreground_bias_rad':float(values.mean())}

def evaluate(condition,arm):
    folder=OUT/condition/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    cached=torch.load(OUT/(condition+'_cache.pt'),weights_only=False);data=cached['data'];records=cached['records']
    neural=arm in NEURAL;model=None
    if neural:
        model=s.model_for(arm).cuda().eval();ckpt=torch.load(s.OUT/'learning'/arm/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model'])
    else:cfg=json.loads((OUT/'protocol.json').read_text())['classical_configs'][arm]
    rows=[];preds=[];t0=time.time();bs=4 if neural else 16
    with torch.inference_mode():
        for off in range(0,len(records),bs):
            b=[t[off:off+bs].cuda() for t in data]
            if neural:
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
            else:pred,_=classical.reconstruct(b,arm,cfg)
            assert torch.isfinite(pred).all()
            for j,row in enumerate(records[off:off+bs]):
                v=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1])
                metrics=dict(zip(s.METRIC_NAMES,map(float,v)));metrics.update(phase_metrics(pred[j:j+1,1:2],b[7][j:j+1,1:2]))
                rows.append({**row,'metrics':metrics})
            preds.append(pred.cpu())
            if not neural:print(condition,arm,off+len(pred),'/',len(records),round(time.time()-t0,1),'s',flush=True)
    s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':torch.cat(preds)})
    s.dump(folder/'per_observation_metrics.json',rows)
    # Here each content has exactly two draws, so the existing equal-category helper is equivalent.
    macro={k:float(np.mean([np.mean([r['metrics'][k] for r in rows if r['cell_type']==c]) for c in sorted({r['cell_type'] for r in rows})])) for k in rows[0]['metrics']}
    s.dump(folder/'complete.json',{'status':'complete','macro':macro,'seconds':time.time()-t0,'observations':len(rows),'source_contents':len({r['content_id'] for r in rows})})
    print('COMPLETE',condition,arm,'RMSE',macro['phase_foreground_rmse_rad'],'SSIM',macro['phase_foreground_ssim'],flush=True)
    del model;torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;calibrated.register();prepare()
    if args.prepare_only:return
    assert (OUT/'preparation_complete.json').exists()
    for condition in CONDITIONS:
        for arm in NEURAL:evaluate(condition,arm)
    for condition in CONDITIONS:
        for arm in CLASSICAL[condition]:evaluate(condition,arm)
    s.dump(OUT/'all_complete.json',{'status':'complete','neural_fits':0,'evaluations':sum(len(NEURAL)+len(CLASSICAL[c]) for c in CONDITIONS)})
if __name__=='__main__':main()
