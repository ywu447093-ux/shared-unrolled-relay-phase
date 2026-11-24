"""Read-only reviewer diagnostics; no retraining or checkpoint selection."""
from pathlib import Path
import sys,json,csv,math,argparse
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/experiments'))
import suite as s
parser=argparse.ArgumentParser();parser.add_argument('--assets-root',type=Path)
args=parser.parse_args()
if args.assets_root:s.OLD=args.assets_root.resolve()
OUT=ROOT/'outputs/reviewer_revision_evidence';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
data,split,records,_=s.load_data(False)
ids=split['holdout'];truth=data[7][ids,1].numpy();sup=data[5][ids,0].numpy();pp=data[2][ids].numpy()
fg=truth>np.maximum(.05*truth.max(axis=(-2,-1),keepdims=True),.03)
summary=json.loads((ROOT/'outputs/single_seed_experiments/summary.json').read_text())
methods=list(summary['results']);predictions={};raw=[];source=[];aggregate={}
def boot(groups,seed=20261103):
 rng=np.random.default_rng(seed);cats=sorted({k[0] for k in groups})
 draws=[]
 for cat in cats:
  values=np.array([v for k,v in sorted(groups.items()) if k[0]==cat])
  draws.append(values[rng.integers(len(values),size=(2000,len(values)))].mean(1))
 return list(map(float,np.quantile(np.mean(draws,axis=0),[.025,.975])))
for method in methods:
 folder=s.OUT/('learning' if method in summary['verification'] else 'classical')/method
 archive=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=True)
 assert archive['names']==[r['name'] for r in records['holdout']]
 pred=archive['predictions'][:,1].numpy();predictions[method]=pred
 error=pred-truth;den=fg.sum((-2,-1));assert np.all(den>0)
 values={'phase_foreground_mae_rad':(abs(error)*fg).sum((-2,-1))/den,
  'phase_foreground_rmse_rad':np.sqrt((error**2*fg).sum((-2,-1))/den),
  'phase_foreground_bias_rad':(error*fg).sum((-2,-1))/den}
 grouped={key:defaultdict(list) for key in values}
 for i,r in enumerate(records['holdout']):
  raw.append({'method':method,'name':r['name'],'category':r['cell_type'],'source_content':r['content_id'],**{k:float(v[i]) for k,v in values.items()}})
  for k,v in values.items():grouped[k][(r['cell_type'],r['content_id'])].append(float(v[i]))
 grouped={k:{g:float(np.mean(v)) for g,v in groups.items()} for k,groups in grouped.items()}
 categories=sorted({key[0] for key in grouped[next(iter(grouped))]})
 macros={k:float(np.mean([np.mean([v for g,v in groups.items() if g[0]==c]) for c in categories])) for k,groups in grouped.items()}
 aggregate[method]={'macro':macros,'ci95':{k:boot(groups) for k,groups in grouped.items()},
  'by_category':{c:{k:float(np.mean([v for g,v in groups.items() if g[0]==c])) for k,groups in grouped.items()} for c in categories}}
 for key in sorted(grouped[next(iter(grouped))]):
  source.append({'method':method,'category':key[0],'source_content':key[1],**{k:groups[key] for k,groups in grouped.items()}})
for method in methods:
 a={(r['category'],r['source_content']):r['phase_foreground_rmse_rad'] for r in source if r['method']==method}
 b={(r['category'],r['source_content']):r['phase_foreground_rmse_rad'] for r in source if r['method']=='stage4'}
 dif={k:a[k]-b[k] for k in a}
 aggregate[method]['rmse_difference_from_stage4']={'macro':aggregate[method]['macro']['phase_foreground_rmse_rad']-aggregate['stage4']['macro']['phase_foreground_rmse_rad'],'conditional_ci95':boot(dif)}
def csv_out(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
csv_out('phase_errors_per_observation.csv',raw);csv_out('phase_errors_per_source.csv',source)
csv_out('phase_error_summary.csv',[{'method':m,**a['macro'],**{'rmse_ci_low':a['ci95']['phase_foreground_rmse_rad'][0],'rmse_ci_high':a['ci95']['phase_foreground_rmse_rad'][1]}} for m,a in aggregate.items()])
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)
# Profiles use the same fixed first example per category as the earlier figure.
# The row is fixed using observed support before examining any method output.
examples=json.loads((ROOT/'outputs/single_seed_experiments/figure_records.json').read_text())
names=[r['name'] for r in records['holdout']]
chosen=[names.index(name) for name in examples['fixed_examples']]
fig,axes=plt.subplots(5,2,figsize=(8.4,9),gridspec_kw={'width_ratios':[1,3.1]})
curves=[];profile_records=[]
line_methods=['HIO_ER','calibrated_unet','amplitude_only','stage4']
labels={'HIO_ER':'HIO–ER (adapted)','calibrated_unet':'Direct encoder–decoder','amplitude_only':'Amplitude only','stage4':'Shared 4 stages'}
colors={'HIO_ER':'#009E73','calibrated_unet':'#E69F00','amplitude_only':'#CC79A7','stage4':'#0072B2'}
for row,i in enumerate(chosen):
 y=int(np.argmax(sup[i].sum(1)))
 dx=float(pp[i,0]*pp[i,1]/pp[i,2]*1e6);x=(np.arange(128)-64)*dx
 axes[row,0].imshow(truth[i],cmap='magma',vmin=0,vmax=math.pi);axes[row,0].axhline(y,color='cyan',lw=.8)
 axes[row,0].set(xticks=[],yticks=[],ylabel=records['holdout'][i]['cell_type'])
 ax=axes[row,1];ax.plot(x,truth[i,y],color='black',lw=1.5,label='Reference')
 for m in line_methods:ax.plot(x,predictions[m][i,y],color=colors[m],lw=1.1,linestyle='--' if m=='amplitude_only' else '-',label=labels[m])
 ax.set(xlim=(x[0],x[-1]),ylim=(-.05,math.pi),ylabel='Phase (rad)');ax.grid(alpha=.18)
 if row<4:ax.tick_params(labelbottom=False)
 else:ax.set_xlabel('Object-plane position (µm)')
 profile_records.append({'name':names[i],'category':records['holdout'][i]['cell_type'],'row_zero_based':y,'object_sampling_um':dx,'row_selection':'maximum observed soft-support row sum; first tie'})
 for j in range(128):curves.append({'name':names[i],'category':records['holdout'][i]['cell_type'],'row_zero_based':y,'column':j,'position_um':float(x[j]),'reference_rad':float(truth[i,y,j]),**{m:float(predictions[m][i,y,j]) for m in line_methods}})
handles,lab=axes[0,1].get_legend_handles_labels();fig.legend(handles,lab,loc='upper center',ncol=3,fontsize=9,frameon=False)
fig.subplots_adjust(top=.91,bottom=.07,hspace=.15,wspace=.15);save(fig,'fixed_phase_profiles');csv_out('fixed_phase_profiles.csv',curves)
# Compact main-text panel uses the first two alphabetical categories; the
# complete five-category figure remains available in the supplement.
fig,axes=plt.subplots(2,2,figsize=(8.4,3.25),gridspec_kw={'width_ratios':[1,3.1]})
for row,i in enumerate(chosen[:2]):
 y=profile_records[row]['row_zero_based'];dx=profile_records[row]['object_sampling_um'];x=(np.arange(128)-64)*dx
 axes[row,0].imshow(truth[i],cmap='magma',vmin=0,vmax=math.pi);axes[row,0].axhline(y,color='cyan',lw=.8)
 axes[row,0].set(xticks=[],yticks=[],ylabel=records['holdout'][i]['cell_type'])
 ax=axes[row,1];ax.plot(x,truth[i,y],color='black',lw=1.5,label='Reference')
 for m in line_methods:ax.plot(x,predictions[m][i,y],color=colors[m],lw=1.1,linestyle='--' if m=='amplitude_only' else '-',label=labels[m])
 ax.set(xlim=(x[0],x[-1]),ylim=(-.05,math.pi),ylabel='Phase (rad)');ax.grid(alpha=.18)
 if row==0:ax.tick_params(labelbottom=False)
 else:ax.set_xlabel('Object-plane position (µm)')
handles,lab=axes[0,1].get_legend_handles_labels();fig.legend(handles,lab,loc='upper center',ncol=3,fontsize=8.5,frameon=False)
fig.subplots_adjust(top=.82,bottom=.17,hspace=.18,wspace=.15);save(fig,'fixed_phase_profiles_main')
# Explicitly post hoc difficult-case visualization, based on source-averaged RMSE.
worst=max((r for r in source if r['method']=='stage4'),key=lambda r:r['phase_foreground_rmse_rad'])
wi=next(i for i,r in enumerate(records['holdout']) if r['content_id']==worst['source_content'])
fig,axes=plt.subplots(1,5,figsize=(8.5,2.3));imgs=[truth[wi],predictions['stage4'][wi],predictions['calibrated_unet'][wi],predictions['amplitude_only'][wi],abs(predictions['stage4'][wi]-truth[wi])]
for j,(ax,img,label) in enumerate(zip(axes,imgs,['Reference','Shared 4 stages','Direct encoder–decoder','Amplitude only','4-stage error'])):
 im=ax.imshow(img,cmap='magma',vmin=0,vmax=1 if j==4 else math.pi);ax.set(xticks=[],yticks=[]);ax.set_title(label,fontsize=9)
 if j in [0,4]:fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.055,pad=.05,label='rad',extend='max' if j==4 and img.max()>1 else 'neither')
fig.tight_layout();save(fig,'difficult_case')
record={'scope':'post hoc read-only diagnostics on existing predictions; no model fitting, tuning or new test cohort',
 'definition':'foreground per-observation absolute error, root mean square error and signed bias in radians; then mean optical realizations per source and mean categories equally',
 'foreground':'reference phase > max(0.05*image maximum,0.03 rad); no offset fitting, registration or rescaling',
 'bootstrap':'2000 stratified source-content resamples conditional on fitted models; not seed variability',
 'results':aggregate,'profiles':profile_records,'difficult_case':{**worst,'display_name':names[wi],'selection':'largest source-averaged four-stage foreground RMSE; first optical realization'},
 'category_source_counts':{c:len({r['content_id'] for r in records['holdout'] if r['cell_type']==c}) for c in sorted({r['cell_type'] for r in records['holdout']})}}
(OUT/'reviewer_diagnostics.json').write_text(json.dumps(record,indent=2),encoding='utf8')
primary=['GS','HIO','HIO_ER','NCPP','calibrated_rescnn','calibrated_unet','calibrated_sisprnet','amplitude_only','stage4']
names_plot=['GS adapted','HIO adapted','HIO–ER adapted','NCPP adapted','Direct residual CNN','Direct encoder–decoder','SiSPRNet adapted','Amplitude only','Shared 4 stages']
fig,axes=plt.subplots(1,3,figsize=(9.5,4.2),sharey=True)
yy=np.arange(len(primary))
for j,key in enumerate(['phase_foreground_ssim','phase_foreground_rmse_rad']):
 vals=np.array([summary['results'][m]['macro'][key] if j==0 else aggregate[m]['macro'][key] for m in primary])
 limits=np.array([summary['results'][m]['ci95'][key] if j==0 else aggregate[m]['ci95'][key] for m in primary])
 axes[j].errorbar(vals,yy,xerr=[vals-limits[:,0],limits[:,1]-vals],fmt='o',capsize=3,color='#0072B2')
 axes[j].set_xlabel('Foreground phase SSIM' if j==0 else 'Foreground phase RMSE (rad)')
axes[0].set_yticks(yy,names_plot,fontsize=9);axes[0].invert_yaxis()
axes[2].plot([summary['results'][m]['latency_ms']['median'] for m in primary],yy,'o',color='#D55E00');axes[2].set(xscale='log',xlabel='Reconstruction time (ms)')
for ax in axes:ax.grid(alpha=.18)
fig.tight_layout();save(fig,'primary_comparison')
for m in ['stage3','stage4','calibrated_unet','amplitude_only','HIO_ER','NCPP','PhysenNet']:
 print(m,json.dumps(aggregate[m]['macro']),flush=True)
print('Difficult case',names[wi],worst['phase_foreground_rmse_rad'],flush=True)
