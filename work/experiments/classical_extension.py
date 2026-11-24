"""Validation-only budget extension when classical selection hits its limit.

Preserve the original search and predictions; never select using evaluation
quality. This is run after every training/fitting job, before final analysis.
"""
from pathlib import Path
import json,shutil,time
import numpy as np
import torch
import suite as s
import classical as c

def same_except(a,b,key):return {k:v for k,v in a.items() if k!=key}=={k:v for k,v in b.items() if k!=key}

def main():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data,split,records,_=s.load_data(False)
 root=s.OUT/'classical';curves={}
 for method in ['GS','HIO','HIO_ER']:
  folder=root/method;extra=folder/'convergence_extension';extra.mkdir(exist_ok=True)
  backup=folder/'initial_search';backup.mkdir(exist_ok=True)
  for name in ['selection.json','complete.json','predictions.pt','holdout_metrics.json','trajectory.json']:
   if not (backup/name).exists():shutil.copy2(folder/name,backup/name)
  original_selection=json.loads((backup/'selection.json').read_text())
  cfg=original_selection['selected'];entries=[]
  for path in folder.glob('validation_*.json'):
   row=json.loads(path.read_text())
   if same_except(row['config'],cfg,'iterations'):entries.append(row)
  assert entries
  if cfg['iterations']==1000:
   for budget in [3000,10000]:
    path=extra/f'validation_{budget:05d}.json'
    if path.exists():entry=json.loads(path.read_text())
    else:
     candidate={**cfg,'iterations':budget}
     mm,rows,_,seconds=c.evaluate(data,split['validation'],records['validation'],method,candidate)
     entry={'config':candidate,'macro':mm,'score':s.score(mm),'seconds':seconds,'per_sample':rows}
     s.dump(path,entry);print('Extended validation',method,budget,mm['phase_foreground_ssim'],flush=True)
    entries.append(entry)
  best=max(entries,key=lambda row:row['score'])
  curves[method]=[{'propagation_pairs':row['config']['iterations'],'foreground_ssim':row['macro']['phase_foreground_ssim'],
    'gradient_correlation':row['macro']['phase_gradient_correlation'],'selection_score':row['score'],
    'selected':row['config']==best['config']} for row in sorted(entries,key=lambda row:row['config']['iterations'])]
  if (extra/'complete.json').exists():continue
  if best['config']!=cfg:
   macro,rows,predictions,seconds=c.evaluate(data,split['holdout'],records['holdout'],method,best['config'],True)
   s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':predictions})
   s.dump(folder/'holdout_metrics.json',rows)
   categories=sorted({r['cell_type'] for r in records['holdout']})
   chosen=[next(i for i,r in enumerate(records['holdout']) if r['cell_type']==category) for category in categories]
   bb=[tensor[[split['holdout'][i] for i in chosen]].cuda() for tensor in data]
   _,trace=c.reconstruct(bb,method,best['config'],True)
   if not trace or trace[-1]['propagation_pairs']!=best['config']['iterations']:
    trace.append({'propagation_pairs':best['config']['iterations'],'metrics':[rows[i]['metrics'] for i in chosen]})
   s.dump(folder/'trajectory.json',{'names':[records['holdout'][i]['name'] for i in chosen],'categories':categories,'records':trace})
   single=[tensor[:1] for tensor in bb];timings=[]
   for repeat in range(7):
    torch.cuda.synchronize();start=time.perf_counter();c.reconstruct(single,method,best['config']);torch.cuda.synchronize()
    if repeat>=2:timings.append((time.perf_counter()-start)*1000)
   done={'status':'complete','config':best['config'],'macro':macro,'evaluation_s':seconds,
    'latency_ms':{'median':float(np.median(timings)),'p10':float(np.quantile(timings,.1)),'p90':float(np.quantile(timings,.9))},
    'method_note':'paired-observation adaptation; validation-only extended iteration budget',
    'script_sha256':s.sha(c.__file__),'extension_sha256':s.sha(__file__)}
   s.dump(folder/'complete.json',done)
  s.dump(folder/'selection.json',{'selected':best['config'],'validation_score':best['score'],
   'rule':'global validation selection; upper-limit original budgets extended to 3000 and 10000, constraints fixed',
   'original_search':'initial_search/'})
  s.dump(extra/'complete.json',{'status':'complete','selected':best['config'],
   'trigger':'original validation selection reached the 1000-iteration upper bound','evaluation_targets_select_settings':False,
   'extension_sha256':s.sha(__file__)})
 # NCPP already chose the shorter of its two tested continuation schedules.
 folder=root/'NCPP';cfg=json.loads((folder/'complete.json').read_text())['config'];entries=[]
 for path in folder.glob('validation_*.json'):
  row=json.loads(path.read_text())
  if same_except(row['config'],cfg,'inner'):entries.append(row)
 curves['NCPP']=[{'propagation_pairs':66*row['config']['inner'],'foreground_ssim':row['macro']['phase_foreground_ssim'],
   'gradient_correlation':row['macro']['phase_gradient_correlation'],'selection_score':row['score'],
   'selected':row['config']==cfg} for row in sorted(entries,key=lambda row:row['config']['inner'])]
 s.dump(s.OUT/'classical_validation_budget_curves.json',curves)

if __name__=='__main__':main()
