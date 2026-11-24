"""Independent dataset and internal prediction checks without refitting."""
import json,itertools
from pathlib import Path
import numpy as np,torch
import suite as s
import reviewer_validation as v

def main():
    torch.set_num_threads(2)
    split=json.loads((s.OLD/'outputs/R1_controlled_experiments_v2/source_split.json').read_text(encoding='utf8'))
    secondary=json.loads((v.OUT/'secondary_source_records.json').read_text(encoding='utf8'))
    partitions={k:{r['content_id'] for r in rows} for k,rows in split.items()};partitions['secondary']={r['content_id'] for r in secondary}
    for a,b in itertools.combinations(partitions,2):assert not partitions[a]&partitions[b],(a,b)
    for name,expected in [('train','train'),('validation','validation')]:
        cache=torch.load(s.ROOT/'work/reviewer_training_v1'/(name+'_cache.pt'),map_location='cpu',weights_only=False)
        assert {r['content_id'] for r in cache['records']}==partitions[expected]
        assert len(cache['records'])==len(partitions[expected])*6
        count=len(partitions[expected]);phase=cache['data'][7][:,1].reshape(count,2,3,128,128)
        geometry=cache['data'][2][:,:3].reshape(count,2,3,3)
        assert torch.equal(phase[:,:,0],phase[:,:,1]) and torch.equal(phase[:,:,0],phase[:,:,2])
        assert torch.equal(geometry[:,:,0],geometry[:,:,1]) and torch.equal(geometry[:,:,0],geometry[:,:,2])
        offset=10000 if name=='train' else 20000
        for r in cache['records']:assert r['field_seed']==2026110400+10*(r['source_index']+offset)+r['simulation_index']
    data,indices,records,_=s.load_data(False);truth=data[7][indices['holdout'],1].numpy().astype(np.float64)
    mask=truth>np.maximum(.05*truth.max(axis=(-2,-1),keepdims=True),.03);diffs={};newrows={}
    for folder,arm in [('fouriergs_v1','fouriergs_paired'),('no_diffraction_v1','stage4_no_diffraction')]:
        path=s.ROOT/'work'/folder/'learning'/arm
        p=torch.load(path/'predictions.pt',map_location='cpu',weights_only=False);rows=json.loads((path/'holdout_metrics_enriched.json').read_text(encoding='utf8'))
        assert p['names']==[r['name'] for r in records['holdout']]==[r['name'] for r in rows]
        err=p['predictions'][:,1].numpy().astype(np.float64)-truth
        metrics={'phase_foreground_rmse_rad':np.sqrt((err**2*mask).sum((-2,-1))/mask.sum((-2,-1))),
          'phase_foreground_mae_rad':(np.abs(err)*mask).sum((-2,-1))/mask.sum((-2,-1))}
        maximum=max(float(np.max(np.abs(values-np.array([r['metrics'][key] for r in rows])))) for key,values in metrics.items())
        assert maximum<5e-7,(arm,maximum);diffs[arm]=maximum
        newrows[arm]=rows
    originalpath=s.OUT/'learning/stage4'
    base=json.loads((originalpath/'holdout_metrics.json').read_text(encoding='utf8'))
    pred=torch.load(originalpath/'predictions.pt',map_location='cpu',weights_only=False)['predictions'][:,1].numpy().astype(np.float64)
    base_rmse=np.sqrt(((pred-truth)**2*mask).sum((-2,-1))/mask.sum((-2,-1)))
    for r,x in zip(base,base_rmse):r['metrics']['phase_foreground_rmse_rad']=float(x)
    def group(rows,key):
        g={}
        for r in rows:g.setdefault((r['cell_type'],r['content_id']),[]).append(r['metrics'][key])
        return {k:float(np.mean(x)) for k,x in sorted(g.items())}
    paired={}
    for arm,rows in newrows.items():
        paired[arm]={}
        for key in ['phase_foreground_ssim','phase_foreground_rmse_rad']:
            aa=group(rows,key);bb=group(base,key);assert aa.keys()==bb.keys()
            rng=np.random.default_rng(20261104);draws=np.zeros(2000);mean=0.
            for cat in sorted({k[0] for k in aa}):
                x=np.array([aa[k]-bb[k] for k in aa if k[0]==cat]);ix=rng.integers(len(x),size=(2000,len(x)))
                mean+=float(x.mean())/5;draws+=x[ix].mean(1)/5
            paired[arm][key]={'mean':mean,'ci95':np.quantile(draws,[.025,.975]).tolist()}
    s.dump(s.ROOT/'work/evidence_analysis/internal_paired_new_controls.json',paired)
    result={'status':'passed','content_counts':{k:len(x) for k,x in partitions.items()},'all_four_source_partitions_disjoint':True,'corrective_training_uses_only_original_train_and_validation':True,'corrective_conditions_share_exact_phase_and_geometry':True,'field_seeds_match_per_observation_records':True,'independent_numpy_internal_metric_max_difference_rad':diffs}
    s.dump(s.ROOT/'work/evidence_analysis/training_audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
