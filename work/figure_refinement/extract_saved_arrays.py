"""Read saved numerical evidence only. No fitting, inference or metric changes."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'work/experiments'));import suite as s
out={};meta={};source=[]
def load(p):
    source.append(str(p.relative_to(R)))
    return torch.load(p,map_location='cpu',weights_only=False)
e=load(R/'outputs/single_seed_experiments/example_inputs.pt')
out['truth']=e['references'][:,1].numpy();out['amplitude']=e['references'][:,0].numpy()
out['obs_sps']=e['observations'][0].numpy();out['obs_amp']=e['observations'][3].numpy();out['obs_support']=e['observations'][5].numpy()
out['geometry']=e['observations'][2].numpy();meta['names']=e['names'];meta['categories']=e['categories'];meta['selection']=e['selection']
full={}
for m in ['HIO_ER','GS','NCPP','calibrated_unet','calibrated_sisprnet','stage4','amplitude_only','fouriergs_paired']:
    folder=R/'work/fouriergs_v1/learning/fouriergs_paired' if m=='fouriergs_paired' else s.OUT/('classical' if m in ['HIO_ER','GS','NCPP'] else 'learning')/m
    a=load(folder/'predictions.pt');ix={n:i for i,n in enumerate(a['names'])}
    out['pred_'+m]=a['predictions'][[ix[n] for n in e['names']],1].numpy()
    if m in ['stage4','calibrated_unet','amplitude_only']:full[m]=a
d=json.loads((R/'outputs/reviewer_revision_evidence/reviewer_diagnostics.json').read_text())
name=d['difficult_case']['display_name'];wi=full['stage4']['names'].index(name)
data,split,records,_=s.load_data(False)
out['worst_truth']=data[7][split['holdout'][wi],1].numpy()
for m,a in full.items():out['worst_'+m]=a['predictions'][a['names'].index(name),1].numpy()
meta['difficult_case']=d['difficult_case'];del data,full
conditions=['fresh_coupled','independent_absorption','pure_phase'];mu=[];condition_records={}
sel=json.loads((R/'work/evidence_analysis/figures/qualitative_selection.json').read_text())
for row,c in enumerate(conditions):
    folder=R/('work/balanced_pairing_v1' if c=='independent_absorption' else 'work/reviewer_validation_v1')
    cache=load(folder/(c+'_cache.pt'));records=cache['records']
    ix=min(range(len(records)),key=lambda i:(records[i]['cell_type'],records[i]['source_qpm'],records[i]['simulation_index']))
    assert records[ix]['content_id']==sel['phase_source']['content_id']
    data=cache['data'];b=[t[ix:ix+1] for t in data];pp=b[2]
    calibrated=(b[0]*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/s._camera_modulus_gain(pp)
    scale=(pp[:,0:1]*pp[:,1:2]/pp[:,2:3].square())[:,:,None,None]*128**2
    out[f'condition_{row}_screen']=(torch.log1p(100*calibrated/scale)/math.log(101))[0,0].numpy()
    out[f'condition_{row}_truth']=data[7][ix,1].numpy()
    out[f'condition_{row}_absorption']=(1-data[3][ix,0]/data[4][ix,0].clamp_min(1e-4)).numpy()
    out[f'condition_{row}_mask']=(data[4][ix,0]>=.12*data[4][ix,0].max()).numpy()
    out[f'condition_{row}_geometry']=data[2][ix].numpy()
    original=R/'work/balanced_pairing_v1/stage4/predictions.pt' if c=='independent_absorption' else R/'work/reviewer_validation_v1'/c/'stage4/predictions.pt'
    mixed=R/'work/balanced_pairing_v1/stage4_mixed/predictions.pt' if c=='independent_absorption' else R/'work/reviewer_training_v1/evaluation'/c/'stage4_mixed/predictions.pt'
    out[f'condition_{row}_original']=load(original)['predictions'][ix,1].numpy()
    out[f'condition_{row}_mixed']=load(mixed)['predictions'][ix,1].numpy()
    condition_records[c]=records[ix];del cache,data
meta['condition_selection']=condition_records;meta['source_files_read']=source
np.savez_compressed(W/'saved_arrays.npz',**out)
(W/'saved_arrays_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
print('Read unchanged saved arrays:',len(out),'arrays; no inference or training.',flush=True)
