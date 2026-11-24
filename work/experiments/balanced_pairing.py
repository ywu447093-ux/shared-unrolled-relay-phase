"""Broaden the fixed-donor diagnostic using every secondary absorption source once."""
from pathlib import Path
import argparse,json,time
from collections import Counter
import numpy as np,torch
from scipy.optimize import linear_sum_assignment
import suite as s
import calibrated
import reviewer_validation as v
import classical

OUT=s.ROOT/'work/balanced_pairing_v1';OUT.mkdir(exist_ok=True)
METHODS=[*v.NEURAL,'stage4_mixed','stage4_coupled','unet_mixed','unet_coupled','fouriergs_paired','stage4_no_diffraction','HIO','NCPP']

def prepare():
    if (OUT/'preparation_complete.json').exists():
        assert json.loads((OUT/'protocol.json').read_text(encoding='utf8'))['code_sha256']==s.sha(__file__)
        return
    prior=torch.load(v.OUT/'fresh_coupled_cache.pt',map_location='cpu',weights_only=False)
    unique=sorted({r['source_qpm']:r['cell_type'] for r in prior['records']}.items())
    cats=np.array([c for p,c in unique]);rng=np.random.default_rng(2026110451)
    cost=rng.uniform(0,1,(105,105));cost[cats[:,None]==cats[None,:]]=1e6
    row,col=linear_sum_assignment(cost);assert np.array_equal(row,np.arange(105))
    assert len(set(col))==105 and all(cats[i]!=cats[j] for i,j in zip(row,col))
    initial=json.loads((v.OUT/'independent_absorption/stage4/complete.json').read_text(encoding='utf8'))['macro']['phase_foreground_rmse_rad']
    protocol={'status':'written before any balanced-pairing reconstruction score',
      'reason':'Initial path-ordered next-category pairing reuses five donor morphologies. Retain those records and broaden final assessment to all105 donor sources.',
      'initial_fixed_donor_stage4_rmse_rad':initial,'initial_donor_contents':5,
      'code_sha256':s.sha(__file__),'original_generator_sha256':s.sha(v.__file__),
      'matching':'Hungarian minimum-cost assignment with seed2026110451 uniform random costs, same-category cost1e6. Every source used exactly once as phase and once as absorption donor; no same-category pair.',
      'sources':105,'realizations':2,'fields':'Same phase, illumination, absorption parameters and optical draws as the frozen fresh coupled control; all changed measurements regenerated.',
      'selection':'No model selection or fitting in this assessment. Additional-training protocol already fixed. Final checkpoints chosen using original-source validation before balanced evaluation.',
      'intervals':'Conditional source-content resampling with donor assignment fixed; not uncertainty across alternative pairings or training seeds.',
      'pairs':[{'source_qpm':unique[i][0],'phase_category':unique[i][1],'donor_qpm':unique[j][0],'donor_category':unique[j][1]} for i,j in zip(row,col)]}
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==protocol
    else:s.dump(p,protocol)
    qs=[]
    for i,(path,c) in enumerate(unique):
        crop=v.gen.content_aware_native_crop(v.gen._read_qpm(path),128,np.random.default_rng(20261104+i))
        qs.append(v.gen.qpm_to_phase_fraction(crop))
    samples=[];records=[]
    for i,(path,c) in enumerate(unique):
        for rep in range(2):
            values,meta=v.pack_sample(qs[i],qs[col[i]],i,rep,'independent_absorption');samples.append(values)
            original=prior['records'][2*i+rep]
            records.append({**original,'name':f'independent_absorption_{i:04d}_{rep}','donor_qpm':unique[col[i]][0],'donor_category':unique[col[i]][1],'pairing':'balanced_one_to_one',**meta})
    data=[torch.stack([x[k] for x in samples]) for k in range(8)]
    assert torch.equal(data[7][:,1],prior['data'][7][:,1]) and torch.equal(data[2][:,:3],prior['data'][2][:,:3])
    assert len({r['donor_qpm'] for r in records})==105
    s.save(OUT/'independent_absorption_cache.pt',{'data':data,'records':records})
    s.dump(OUT/'preparation_complete.json',{'status':'complete','phase_sources':105,'donor_sources':105,'observations':210,'matched_phase_and_geometry':True})
    print('PREPARED balanced one-to-one absorption pairing:105 phase and105 donor contents',flush=True)

def model_checkpoint(arm):
    if arm=='fouriergs_paired':
        from fouriergs_paired import PairedFourierGSNet
        return PairedFourierGSNet(),s.ROOT/'work/fouriergs_v1/learning'/arm/'best.pt'
    if arm=='stage4_no_diffraction':
        from no_diffraction_train import NoDiffraction
        return NoDiffraction(),s.ROOT/'work/no_diffraction_v1/learning'/arm/'best.pt'
    if arm in ['stage4_mixed','stage4_coupled','unet_mixed','unet_coupled']:
        return s.model_for('stage4' if arm.startswith('stage4') else 'calibrated_unet'),s.ROOT/'work/reviewer_training_v1'/arm/'best.pt'
    return s.model_for(arm),s.OUT/'learning'/arm/'best.pt'

def evaluate(arm):
    folder=OUT/arm;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    cached=torch.load(OUT/'independent_absorption_cache.pt',map_location='cpu',weights_only=False);rows=[];preds=[]
    model=None
    if arm not in ['HIO','NCPP']:
        model,path=model_checkpoint(arm);model=model.cuda().eval();ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);bs=2 if arm=='fouriergs_paired' else 4
    else:cfg=json.loads((v.OUT/'protocol.json').read_text(encoding='utf8'))['classical_configs'][arm];bs=16
    with torch.inference_mode():
        for off in range(0,210,bs):
            b=[t[off:off+bs].cuda() for t in cached['data']]
            if model is not None:
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
            else:pred,_=classical.reconstruct(b,arm,cfg)
            assert torch.isfinite(pred).all()
            for j,r in enumerate(cached['records'][off:off+bs]):
                vv=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1]);mm=dict(zip(s.METRIC_NAMES,map(float,vv)))
                mm.update(v.phase_metrics(pred[j:j+1,1:2],b[7][j:j+1,1:2]));rows.append({**r,'metrics':mm})
            preds.append(pred.cpu())
    macro={k:float(np.mean([np.mean([r['metrics'][k] for r in rows if r['cell_type']==c]) for c in sorted({r['cell_type'] for r in rows})])) for k in rows[0]['metrics']}
    s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':torch.cat(preds)})
    s.dump(folder/'per_observation_metrics.json',rows);s.dump(folder/'complete.json',{'status':'complete','macro':macro,'source_contents':105,'donor_contents':105,'observations':210})
    print('BALANCED',arm,macro['phase_foreground_rmse_rad'],flush=True)
    del model;torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();torch.set_num_threads(2);prepare()
    if args.prepare_only:return
    for name in ['reviewer_training_v1','fouriergs_v1','no_diffraction_v1']:assert (s.ROOT/'work'/name/'all_complete.json').exists()
    calibrated.register();torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for arm in METHODS:evaluate(arm)
    s.dump(OUT/'all_complete.json',{'status':'complete','methods':METHODS,'phase_contents':105,'donor_contents':105})

if __name__=='__main__':main()
