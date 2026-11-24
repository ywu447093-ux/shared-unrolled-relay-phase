"""Read-only source-level analysis of retrospective and object-class tests."""
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,torch
import suite as s
import reviewer_validation as v

DEST=s.ROOT/'work/evidence_analysis';DEST.mkdir(exist_ok=True)
METRICS=['phase_foreground_rmse_rad','phase_foreground_mae_rad','phase_foreground_ssim','amp_pred_ssim']

def grouped(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[(r['cell_type'],r['content_id'])].append(float(r['metrics'][key]))
    assert len(groups)==105 and all(len(x)==2 for x in groups.values())
    return {k:float(np.mean(x)) for k,x in sorted(groups.items())}

def bootstrap(groups):
    cats=sorted({k[0] for k in groups});rng=np.random.default_rng(20261104)
    draws=np.zeros(2000)
    means={}
    for c in cats:
        x=np.array([x for (cat,cid),x in groups.items() if cat==c])
        means[c]=float(x.mean());draws+=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)/len(cats)
    return {'mean':float(np.mean(list(means.values()))),'ci95':np.quantile(draws,[.025,.975]).tolist(),'category_means':means}

def audit_condition(condition,folders,balanced=False):
    root=s.ROOT/'work/balanced_pairing_v1' if balanced else v.OUT
    cache=torch.load(root/(condition+'_cache.pt'),map_location='cpu',weights_only=False)
    truth=cache['data'][7][:,1].numpy().astype(np.float64)
    fg=truth>np.maximum(.05*truth.max(axis=(-2,-1),keepdims=True),.03)
    diagnostics=[]
    for i,r in enumerate(cache['records']):
        support=cache['data'][5][i,0].numpy();flat=cache['data'][4][i,0].numpy();amp=cache['data'][3][i,0].numpy()
        absorption=1-amp/np.maximum(flat,1e-4)
        corr=np.corrcoef(absorption[fg[i]],truth[i][fg[i]])[0,1]
        diagnostics.append({**r,'metrics':{'phase_foreground_outside_support_half':float(np.mean(support[fg[i]]<.5)),
          'phase_absorption_correlation':float(corr),'support_foreground_weight':float(support[fg[i]].mean())}})
    diagnostic={k:bootstrap(grouped(diagnostics,k)) for k in diagnostics[0]['metrics']}
    supportarray=cache['data'][5][:,0].numpy().astype(np.float64)
    flatarray=cache['data'][4][:,0].numpy().astype(np.float64)
    illuminated=flatarray>=.12*flatarray.max((-2,-1),keepdims=True)
    backgroundweight=(1-supportarray)**4*illuminated
    border=np.zeros((128,128));border[:4]=border[-4:]=1;border[:,:4]=border[:,-4:]=1
    for i in range(len(truth)):
        if backgroundweight[i].sum()<1e-8:backgroundweight[i]=illuminated[i]*border
        if backgroundweight[i].sum()<1e-8:backgroundweight[i]=border
    maxdiff=0.;results={};perimage=[];gauge={}
    for method,folder in folders:
        rows=json.loads((folder/'per_observation_metrics.json').read_text())
        pred=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=False)
        assert pred['names']==[r['name'] for r in cache['records']]==[r['name'] for r in rows]
        delta=pred['predictions'][:,1].numpy().astype(np.float64)-truth
        denominator=fg.sum(axis=(-2,-1))
        values={'phase_foreground_mae_rad':(np.abs(delta)*fg).sum(axis=(-2,-1))/denominator,
          'phase_foreground_rmse_rad':np.sqrt((delta**2*fg).sum(axis=(-2,-1))/denominator),
          'phase_foreground_bias_rad':(delta*fg).sum(axis=(-2,-1))/denominator}
        for k,x in values.items():
            old=np.array([r['metrics'][k] for r in rows]);maxdiff=max(maxdiff,float(np.max(np.abs(old-x))))
        results[method]={k:bootstrap(grouped(rows,k)) for k in METRICS}
        phase=pred['predictions'][:,1].numpy().astype(np.float64)
        offset=np.arctan2((np.sin(phase)*backgroundweight).sum((-2,-1)),(np.cos(phase)*backgroundweight).sum((-2,-1)))
        referenced=np.clip((phase-offset[:,None,None]+np.pi)%(2*np.pi)-np.pi,0,np.pi)
        referenced_rmse=np.sqrt(((referenced-truth)**2*fg).sum((-2,-1))/denominator)
        gauge_rows=[{**r,'metrics':{'observed_background_rmse_rad':float(referenced_rmse[i]),'background_offset_rad':float(offset[i])}} for i,r in enumerate(rows)]
        gauge[method]={k:bootstrap(grouped(gauge_rows,k)) for k in gauge_rows[0]['metrics']}
        pairing='balanced_one_to_one105' if balanced else ('initial_five_donors' if condition=='independent_absorption' else 'not_applicable')
        for r in rows:perimage.append({'condition':condition,'pairing':pairing,'method':method,'source_id':r['content_id'],'category':r['cell_type'],'name':r['name'],'absorption_donor_source':r.get('donor_qpm','') if condition=='independent_absorption' else '',**r['metrics']})
    assert maxdiff<5e-7,(condition,maxdiff)
    return results,diagnostic,maxdiff,perimage,gauge

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original-only',action='store_true');args=ap.parse_args()
    balancedroot=s.ROOT/'work/balanced_pairing_v1'
    if not args.original_only:
        for name in ['reviewer_training_v1','fouriergs_v1','no_diffraction_v1']:assert (s.ROOT/'work'/name/'all_complete.json').exists()
        subprocess.run([sys.executable,'-X','utf8',str(s.HERE/'balanced_pairing.py')],check=True)
    torch.set_num_threads(2);allresults={};allrows=[];audit={};diagnostics={};rawgroups={};allgauge={}
    for condition in v.CONDITIONS:
        balanced=condition=='independent_absorption' and not args.original_only
        folders=[(a,balancedroot/a if balanced else v.OUT/condition/a) for a in [*v.NEURAL,*v.CLASSICAL[condition]]]
        if not args.original_only:
            for arm in ['stage4_mixed','stage4_coupled','unet_mixed','unet_coupled']:
                folder=balancedroot/arm if balanced else s.ROOT/'work/reviewer_training_v1/evaluation'/condition/arm
                assert (folder/'complete.json').exists(),folder
                folders.append((arm,folder))
            folder=balancedroot/'fouriergs_paired' if balanced else s.ROOT/'work/fouriergs_v1/evaluation'/condition/'fouriergs_paired'
            assert (folder/'complete.json').exists(),folder
            folders.append(('fouriergs_paired',folder))
            folder=balancedroot/'stage4_no_diffraction' if balanced else s.ROOT/'work/no_diffraction_v1/evaluation'/condition/'stage4_no_diffraction'
            assert (folder/'complete.json').exists(),folder
            folders.append(('stage4_no_diffraction',folder))
        results,diagnostic,maxdiff,rows,gauge=audit_condition(condition,folders,balanced)
        allgauge[condition]=gauge
        allresults[condition]=results;diagnostics[condition]=diagnostic;audit[condition]={'independent_numpy_max_absolute_error_rad':maxdiff}
        allrows.extend(rows)
        rawgroups[condition]={m:{k:grouped(json.loads((p/'per_observation_metrics.json').read_text()),k) for k in METRICS} for m,p in folders}
    paired={}
    for condition in v.CONDITIONS:
        groups=rawgroups[condition];comparisons=[('stage4','stage3'),('stage4','no_global'),('stage4','unshared')]
        if not args.original_only:comparisons += [('stage4_mixed','stage4_coupled'),('unet_mixed','unet_coupled'),('stage4_mixed','unet_mixed'),('stage4','fouriergs_paired'),('stage4','stage4_no_diffraction')]
        paired[condition]={}
        for a,b in comparisons:
            paired[condition][a+' minus '+b]={}
            for k in METRICS:
                aa=groups[a][k];bb=groups[b][k];assert aa.keys()==bb.keys()
                paired[condition][a+' minus '+b][k]=bootstrap({x:aa[x]-bb[x] for x in aa})
    paired_conditions={}
    for condition in v.CONDITIONS[2:]:
        paired_conditions[condition+' minus fresh_coupled']={}
        for method in rawgroups[condition].keys()&rawgroups['fresh_coupled'].keys():
            aa=rawgroups[condition][method]['phase_foreground_rmse_rad'];bb=rawgroups['fresh_coupled'][method]['phase_foreground_rmse_rad']
            paired_conditions[condition+' minus fresh_coupled'][method]=bootstrap({x:aa[x]-bb[x] for x in aa})
    final={'results':allresults,'paired_methods':paired,'paired_conditions':paired_conditions,'support_diagnostics':diagnostics,'audit':audit,
      'observed_background_gauge_sensitivity':allgauge,
      'gauge_rule':'Same circular mean of predicted phase over (1-support)^4 with flat>=12% image maximum; illuminated4pixelborder then gridborder fallback; subtract, wrap[-pi,pi],clip[0,pi]. No reference phase used to choose offset. Read-only diagnostic; primary outputs unchanged.',
      'bootstrap':'2000 source-content resamples within each of five categories; fixed fitted models; paired identities reused',
      'sources':105,'observations_per_condition':210,'single_training_seed':101,
      'independent_absorption_pairing':'initial five donors' if args.original_only else 'balanced one-to-one105 donors; assignment fixed before evaluation',
      'initial_fixed_donor_results':'retained separately in original_analysis.json; no original record overwritten'}
    s.dump(DEST/('original_analysis.json' if args.original_only else 'evidence_analysis.json'),final)
    csvpath=DEST/('original_per_observation.csv' if args.original_only else 'per_observation.csv')
    keys=list(allrows[0]);keys+=sorted({k for r in allrows for k in r}-set(keys))
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(allrows)
    print(json.dumps({'audit':audit,'support_diagnostics':diagnostics},indent=2),flush=True)

if __name__=='__main__':main()
