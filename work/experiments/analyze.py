"""Summarize only completed experiments and draw publication figures."""
from pathlib import Path
import argparse,csv,json,math,sys,time,subprocess
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import suite as s
import stability
stability.register()

OUT=s.ROOT/'outputs/single_seed_experiments'
FIG=OUT/'figures'
STAGES=['stage1','stage2','stage3','stage4','stage6','stage8']
MIXERS=['no_global','no_attention','heads1','stage4','heads8']
SHARING=['stage4','private_heads','private_decoders','unshared']
BASELINES=['GS','HIO','HIO_ER','NCPP','PhysenNet','calibrated_rescnn','calibrated_unet','calibrated_sisprnet','amplitude_only','stage4']
SHORT={**s.LABELS,'stage4':'Shared four-stage model','stage1':'1 stage','stage2':'2 stages','stage3':'3 stages','stage6':'6 stages','stage8':'8 stages',
 'no_global':'CNN bottleneck','no_attention':'FFN only','heads1':'1 head','heads8':'8 heads','private_heads':'Private heads',
 'private_decoders':'Private decoders','unshared':'Unshared networks','direct_rescnn':'Residual CNN','direct_unet':'Direct U-Net',
 'sisprnet_adapted':'Normalized-SPS SiSPRNet','amplitude_only':'Amplitude only','GS':'GS adapted','HIO':'HIO adapted','HIO_ER':'HIO–ER adapted','NCPP':'NCPP adapted','PhysenNet':'PhysenNet adapted',
 'calibrated_unet':'Calibrated U-Net','calibrated_rescnn':'Calibrated residual CNN','calibrated_sisprnet':'SiSPRNet calibrated'}
SHORT.update({'stage6_lr1e4':'6 stages, lower LR','stage8_lr1e4':'8 stages, lower LR'})

def grouped(rows,key):
 groups=defaultdict(list)
 for row in rows:groups[(row['cell_type'],row['content_id'])].append(row['metrics'][key])
 return {k:float(np.mean(v)) for k,v in groups.items()}
def bootstrap(groups,seed=20261103):
 keys=sorted(groups);bycat=defaultdict(list)
 for k in keys:bycat[k[0]].append(groups[k])
 rng=np.random.default_rng(seed);draws=np.mean([np.asarray(v)[rng.integers(len(v),size=(2000,len(v)))].mean(1) for v in bycat.values()],axis=0)
 return [float(x) for x in np.quantile(draws,[.025,.975])]

def write_csv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def finish_figure(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight')
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
 plt.close(fig)

def plot_learning_curves():
 fig,axes=plt.subplots(2,2,figsize=(8.6,7.8))
 legend_names={**SHORT,'stage4':'4 stages','calibrated_unet':'U-Net (calibrated)',
  'calibrated_rescnn':'Residual CNN (calibrated)','calibrated_sisprnet':'SiSPRNet (calibrated)',
  'unshared':'Unshared','no_global':'CNN bottleneck'}
 for ax,arms,title in zip(axes.flat,[STAGES,MIXERS,SHARING,['stage4','calibrated_unet','calibrated_rescnn','calibrated_sisprnet','amplitude_only']],['Stage count','Attention','Sharing','Direct reconstruction controls']):
  if arms is STAGES:arms=STAGES+list(stability.MAPPING)
  for arm in arms:
   hist=json.loads((s.OUT/'learning'/arm/'history.json').read_text())
   ax.plot([r['epoch'] for r in hist],[r['validation_macro']['phase_foreground_ssim'] for r in hist],linestyle='--' if '_lr' in arm else '-',label=legend_names[arm])
  ax.set(xlabel='Training epoch',ylabel='Validation phase SSIM',title=title)
  ax.tick_params(labelsize=10);ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10);ax.title.set_size(11)
  ax.legend(fontsize=8.4,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.26),frameon=False,columnspacing=1.1,handlelength=2)
 fig.subplots_adjust(left=.1,right=.98,top=.94,bottom=.18,hspace=1.02,wspace=.33)
 finish_figure(fig,'learning_curves')

def benchmark_all():
 data,split,records,_=s.load_data(False);b=[t[split['holdout'][:1]].cuda() for t in data]
 result={}
 for arm in s.ARMS:
  path=s.OUT/'learning'/arm/'timing.json'
  if path.exists():result[arm]=json.loads(path.read_text());continue
  model=s.model_for(arm).cuda().eval();ck=torch.load(s.OUT/'learning'/arm/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
  timings=[];torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
   for i in range(35):
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):model(b)
    torch.cuda.synchronize()
    if i>=5:timings.append((time.perf_counter()-start)*1000)
  info={'median':float(np.median(timings)),'p10':float(np.quantile(timings,.1)),'p90':float(np.quantile(timings,.9)),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'batch':1,'repeats':30,'warmup':5,'precision':'BF16 neural/FP32 physical'}
  s.dump(path,info);result[arm]=info;del model;torch.cuda.empty_cache()
 return result

def verify_predictions():
 data,split,records,_=s.load_data(False);report={}
 for arm in s.ARMS:
  folder=s.OUT/'learning'/arm;check=folder/'reproduction.json'
  if check.exists():report[arm]=json.loads(check.read_text());continue
  model=s.model_for(arm).cuda().eval();ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
  macro,rows,pred=s.evaluate(model,data,split['holdout'],records['holdout'],True)
  archive=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=True)
  error=float((pred-archive['predictions']).abs().max())
  assert archive['names']==[r['name'] for r in rows] and error<=1e-5,(arm,error)
  reference=json.loads((folder/'complete.json').read_text())['macro']
  metric_error=max(abs(macro[k]-reference[k]) for k in macro);assert metric_error<1e-6
  info={'maximum_prediction_difference':error,'maximum_metric_difference':metric_error,'samples':128,'checkpoint_sha256':s.sha(folder/'best.pt')}
  s.dump(check,info);report[arm]=info;del model;torch.cuda.empty_cache()
 return report

def verify_stored_metrics():
 """Recompute per-observation metrics from raw arrays for every method."""
 data,split,records,_=s.load_data(False);report={}
 for arm in s.ARMS+['GS','HIO','HIO_ER','NCPP','PhysenNet']:
  folder=s.OUT/('learning' if arm in s.ARMS else 'classical')/arm
  archive=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=True)
  rows=json.loads((folder/'holdout_metrics.json').read_text())
  assert archive['names']==[r['name'] for r in records['holdout']]==[r['name'] for r in rows]
  pred=archive['predictions'];assert tuple(pred.shape)==(128,2,128,128) and torch.isfinite(pred).all()
  maximum=0.
  for i,idx in enumerate(split['holdout']):
   args=[pred[i:i+1],data[7][idx:idx+1],data[5][idx:idx+1],data[3][idx:idx+1],data[4][idx:idx+1]]
   # Match the metric device used when the original record was produced.
   if arm!='PhysenNet':args=[x.cuda() for x in args]
   values=s.compute_validation_metrics(*args)
   maximum=max(maximum,max(abs(float(v)-rows[i]['metrics'][k]) for k,v in zip(s.METRIC_NAMES,values)))
  assert maximum<1e-6,(arm,maximum)
  report[arm]={'samples':128,'maximum_per_observation_metric_difference':maximum}
 s.dump(s.OUT/'raw_metric_verification.json',report)
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
  'svg.fonttype':'none','pdf.fonttype':42,'axes.grid':True,'grid.alpha':.18})
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
 if args.partial:
  fig,ax=plt.subplots(figsize=(7,4))
  for arm in s.ARMS:
   path=s.OUT/'learning'/arm/'history.json'
   if path.exists():
    hist=json.loads(path.read_text());ax.plot([r['epoch'] for r in hist],[r['validation_macro']['phase_foreground_ssim'] for r in hist],label=SHORT[arm])
  ax.set(xlabel='Training epoch',ylabel='Validation foreground phase SSIM',title='Ongoing single-seed experiment')
  ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(s.OUT/'ongoing_training.png',dpi=150);return
 for arm in s.ARMS:assert (s.OUT/'learning'/arm/'complete.json').exists(),f'Incomplete: {arm}'
 for method in ['GS','HIO','HIO_ER','NCPP','PhysenNet']:assert (s.OUT/'classical'/method/'complete.json').exists(),f'Incomplete: {method}'
 diagnostics={}
 for arm in ['stage6','stage8']:
  hist=json.loads((s.OUT/'learning'/arm/'history.json').read_text())
  values=[r['validation_macro']['phase_foreground_ssim'] for r in hist]
  if max(values)-values[-1]>.2:
   with (s.OUT/f'{arm}_final_diagnostic.log').open('w',encoding='utf8') as log:
    subprocess.run([sys.executable,str(s.HERE/'inspect_deep_training.py'),'--arm',arm,'--assets-root',str(s.OLD)],stdout=log,stderr=subprocess.STDOUT,check=True)
   diagnostics[arm]=json.loads((s.OUT/'learning'/arm/'training_state_diagnostic.json').read_text())
 import classical_extension
 classical_extension.main()
 timing=benchmark_all();verification=verify_predictions();raw_verification=verify_stored_metrics()
 results={};all_samples=[];source_rows=[]
 for arm in s.ARMS+['GS','HIO','HIO_ER','NCPP','PhysenNet']:
  folder=s.OUT/('learning' if arm in s.ARMS else 'classical')/arm
  rows=json.loads((folder/'holdout_metrics.json').read_text());done=json.loads((folder/'complete.json').read_text())
  mm=s.macro(rows)
  assert max(abs(mm[k]-done['macro'][k]) for k in mm)<1e-12
  groups={k:grouped(rows,k) for k in s.METRIC_NAMES};ids=sorted(groups['phase_foreground_ssim'])
  assert len(rows)==128 and len(ids)==64
  cis={k:bootstrap(v) for k,v in groups.items()}
  config=json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else done['config']
  if arm=='PhysenNet':
   from physennet import PhysenUNet
   measured_model=PhysenUNet();config={**config,'parameters':sum(p.numel() for p in measured_model.parameters())};del measured_model
  results[arm]={'label':SHORT[arm],'macro':mm,'ci95':cis,'config':config,
   'latency_ms':timing[arm] if arm in timing else done['latency_ms'],'best_epoch':done.get('best_epoch'),
   'per_source':[{**{'category':key[0],'source_content':key[1]},**{metric:groups[metric][key] for metric in groups}} for key in ids]}
  all_samples.extend([{'method':arm,'name':r['name'],'category':r['cell_type'],'source_content':r['content_id'],**r['metrics']} for r in rows])
  source_rows.extend([{'method':arm,**r} for r in results[arm]['per_source']])
 ref=results['stage4'];paired={}
 for arm,rr in results.items():
  a={r['source_content']:r for r in rr['per_source']};b={r['source_content']:r for r in ref['per_source']}
  differences={(row['category'],key):a[key]['phase_foreground_ssim']-row['phase_foreground_ssim'] for key,row in b.items()}
  cats=sorted({key[0] for key in differences});delta=float(np.mean([np.mean([v for k,v in differences.items() if k[0]==c]) for c in cats]))
  paired[arm]={'foreground_ssim_difference_from_stage4':delta,'conditional_ci95':bootstrap(differences)}
 trajectories={}
 for arm in s.ARMS:
  hist=json.loads((s.OUT/'learning'/arm/'history.json').read_text())
  values=[row['validation_macro']['phase_foreground_ssim'] for row in hist]
  trajectories[arm]={'peak_foreground_ssim':max(values),'peak_foreground_epoch':1+int(np.argmax(values)),
    'final_foreground_ssim':values[-1],'decline_from_peak':max(values)-values[-1],
    'selected_epoch':results[arm]['best_epoch']}
 summary={'status':'completed_single_seed_exploration','seed':101,'epochs':40,'independent_evaluation_contents':64,'evaluation_observations':128,
   'results':results,'paired':paired,'verification':verification,'raw_metric_verification':raw_verification,'interpretation':'single training seed and fixed exploration budget; conditional cohort intervals do not represent training-seed variability',
   'validation_trajectories':trajectories,
   'training_state_diagnostics':diagnostics,
   'best_validation_depth_with_sensitivity':max(STAGES+list(stability.MAPPING),key=lambda a: max(r['selection_score'] for r in json.loads((s.OUT/'learning'/a/'history.json').read_text()))),
   'best_validation_stage':max(STAGES,key=lambda a: max(r['selection_score'] for r in json.loads((s.OUT/'learning'/a/'history.json').read_text())))}
 s.dump(OUT/'summary.json',summary);write_csv(OUT/'per_observation_metrics.csv',all_samples);write_csv(OUT/'per_source_metrics.csv',source_rows)
 curves=json.loads((s.OUT/'classical_validation_budget_curves.json').read_text())
 write_csv(OUT/'classical_validation_budget.csv',[{'method':method,**row} for method,rows in curves.items() for row in rows])
 fig,axes=plt.subplots(1,2,figsize=(8,3.4))
 for method,rows in curves.items():
  for ax,key,label in zip(axes,['foreground_ssim','selection_score'],['Validation foreground phase SSIM','Validation selection score']):
   ax.plot([row['propagation_pairs'] for row in rows],[row[key] for row in rows],'o-',label=method.replace('_','–'))
   ax.set(xscale='log',xlabel='Forward/inverse propagation pairs',ylabel=label)
 axes[0].legend(fontsize=9);fig.tight_layout();finish_figure(fig,'classical_validation_budget')
 condensed=[]
 for arm,r in results.items():condensed.append({'method':arm,'label':r['label'],'parameters':r['config'].get('parameters',''),
   'selected_epoch':r['best_epoch'],'foreground_phase_ssim':r['macro']['phase_foreground_ssim'],'gradient_correlation':r['macro']['phase_gradient_correlation'],
   'amplitude_ssim':r['macro']['amp_pred_ssim'],'latency_ms_median':r['latency_ms']['median'],
   'fg_ssim_difference_from_stage4':paired[arm]['foreground_ssim_difference_from_stage4'],
   'paired_ci95_low':paired[arm]['conditional_ci95'][0],'paired_ci95_high':paired[arm]['conditional_ci95'][1]})
 write_csv(OUT/'comparison_summary.csv',condensed)
 fig,axes=plt.subplots(1,3,figsize=(9,3.2))
 for ax,key,label in zip(axes,['phase_foreground_ssim','phase_gradient_correlation','latency'],['Foreground phase SSIM','Phase-gradient correlation','Single-image inference (ms)']):
  xs=[1,2,3,4,6,8];ys=[results[a]['latency_ms']['median'] if key=='latency' else results[a]['macro'][key] for a in STAGES]
  if key=='latency':ax.plot(xs,ys,'o-',color='#0072B2')
  else:
   ci=np.array([results[a]['ci95'][key] for a in STAGES]);ax.errorbar(xs,ys,yerr=[np.array(ys)-ci[:,0],ci[:,1]-np.array(ys)],fmt='o-',color='#0072B2',capsize=3)
  if key!='latency':
   lower=list(stability.MAPPING);lower_values=np.array([results[a]['macro'][key] for a in lower])
   lower_ci=np.array([results[a]['ci95'][key] for a in lower])
   ax.errorbar([6,8],lower_values,yerr=[lower_values-lower_ci[:,0],lower_ci[:,1]-lower_values],fmt='s--',capsize=3,color='#D55E00',label='Lower LR check')
  ax.set(xlabel='Reconstruction stages',ylabel=label,xticks=xs)
 axes[0].legend(fontsize=9)
 fig.tight_layout();finish_figure(fig,'stage_count_ablation')
 fig,axes=plt.subplots(1,2,figsize=(8,3.8))
 for ax,arms,title,labels in zip(axes,[MIXERS,SHARING],['Global mixing','Weight sharing'],
       [['CNN','FFN only','1 head','4 heads','8 heads'],['Shared','Private\nheads','Private\ndecoders','Unshared']]):
  ys=[results[a]['macro']['phase_foreground_ssim'] for a in arms];ci=np.array([results[a]['ci95']['phase_foreground_ssim'] for a in arms]);xx=np.arange(len(arms))
  ax.errorbar(xx,ys,yerr=[np.array(ys)-ci[:,0],ci[:,1]-np.array(ys)],fmt='o',color='#0072B2',capsize=3)
  ax.set_xticks(xx,labels,rotation=35 if arms==MIXERS else 0,ha='right' if arms==MIXERS else 'center',fontsize=10);ax.set(title=title,ylabel='Foreground phase SSIM')
 fig.tight_layout();finish_figure(fig,'mixing_sharing_baselines')
 fig,axes=plt.subplots(1,2,figsize=(9,4.5),sharey=True)
 yy=np.arange(len(BASELINES));values=[results[a]['macro']['phase_foreground_ssim'] for a in BASELINES]
 ci=np.array([results[a]['ci95']['phase_foreground_ssim'] for a in BASELINES])
 axes[0].errorbar(values,yy,xerr=[np.array(values)-ci[:,0],ci[:,1]-np.array(values)],fmt='o',capsize=3,color='#0072B2')
 axes[0].set_yticks(yy,[SHORT[a] for a in BASELINES],fontsize=10);axes[0].invert_yaxis();axes[0].set_xlabel('Foreground phase SSIM')
 axes[1].plot([results[a]['latency_ms']['median'] for a in BASELINES],yy,'o',color='#D55E00');axes[1].set(xscale='log',xlabel='Reconstruction time (ms)')
 fig.tight_layout();finish_figure(fig,'baseline_quality_cost')
 plot_learning_curves()
 # Fixed first exported example from each class, never chosen by quality.
 data,split,records,_=s.load_data(False);categories=sorted({r['cell_type'] for r in records['holdout']});chosen=[next(i for i,r in enumerate(records['holdout']) if r['cell_type']==c) for c in categories]
 example_ids=[split['holdout'][i] for i in chosen]
 s.save(OUT/'example_inputs.pt',{'names':[records['holdout'][i]['name'] for i in chosen],'categories':categories,
  'observations':[tensor[example_ids] for tensor in data[:7]],'references':data[7][example_ids],
  'selection':'first source example per category; references separate from forward inputs'})
 chosen_methods=['GS','HIO_ER','NCPP','calibrated_unet','calibrated_sisprnet','stage4']
 pred_by={}
 for arm in chosen_methods:
  folder=s.OUT/('learning' if arm in s.ARMS else 'classical')/arm
  pred_by[arm]=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=True)['predictions'][:,1].numpy()
 fig,axes=plt.subplots(5,8,figsize=(9,6.5));error_saturated=False
 for row,index in enumerate(chosen):
  truth=data[7][split['holdout'][index],1].numpy()
  imgs=[truth]+[pred_by[a][index] for a in chosen_methods]+[abs(pred_by['stage4'][index]-truth)]
  error_saturated=error_saturated or float(imgs[-1].max())>.65
  for col,img in enumerate(imgs):
   axes[row,col].imshow(img,cmap='magma',vmin=0,vmax=.65 if col==7 else math.pi);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
   if row==0:axes[row,col].set_title(['Reference','GS','HIO–ER','NCPP','U-Net','SiSPRNet','4 stages','Phase error'][col],fontsize=12)
   if col==0:axes[row,col].set_ylabel(categories[row],fontsize=12)
 fig.tight_layout(rect=[0,.09,1,1])
 from matplotlib.cm import ScalarMappable
 cax=fig.add_axes([.12,.04,.68,.018]);cb=fig.colorbar(ScalarMappable(norm=matplotlib.colors.Normalize(0,math.pi),cmap='magma'),cax=cax,orientation='horizontal');cb.set_label('Phase (rad)',fontsize=11.5);cb.set_ticks([0,math.pi/2,math.pi],labels=['0','π/2','π']);cb.ax.tick_params(labelsize=10.5)
 cax=fig.add_axes([.86,.04,.1,.018]);cb=fig.colorbar(ScalarMappable(norm=matplotlib.colors.Normalize(0,.65),cmap='magma'),cax=cax,orientation='horizontal',extend='max' if error_saturated else 'neither');cb.set_label('Error (rad)',fontsize=11.5);cb.set_ticks([0,.65]);cb.ax.tick_params(labelsize=10.5)
 finish_figure(fig,'baseline_reconstructions')
 s.dump(OUT/'figure_records.json',{'fixed_examples':[records['holdout'][i]['name'] for i in chosen],
  'categories':categories,'phase_range':[0,math.pi],'error_range':[0,.65],'selection':'first original export per category, independent of reconstruction quality'})
 print('Completed analysis of',len(results),'methods',flush=True)
if __name__=='__main__':main()
