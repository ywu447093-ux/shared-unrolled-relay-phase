"""Publication raster figures generated only from completed experiment records."""
import json,math
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import suite as s
import reviewer_validation as v

OUT=s.ROOT/'work/evidence_analysis/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
    'axes.linewidth':.7,'lines.linewidth':.9,'savefig.dpi':600,'figure.facecolor':'white'})
COLORS=['#0072B2','#D55E00','#009E73']

def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18,lw=.5);ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)

def point(ax,x,y,color,marker='o'):
    m=x['mean'];lo,hi=x['ci95'];ax.errorbar(m,y,xerr=np.array([[m-lo],[hi-m]]),fmt=marker,color=color,ms=3.1,capsize=2,lw=.8)

def comparison(a,modern):
    old=json.loads((s.ROOT/'outputs/single_seed_experiments/summary.json').read_text())['results']
    diag=json.loads((s.ROOT/'outputs/reviewer_revision_evidence/reviewer_diagnostics.json').read_text())['results']
    methods=['GS','HIO','HIO_ER','NCPP','calibrated_rescnn','calibrated_unet','calibrated_sisprnet','fouriergs_paired','amplitude_only','stage4']
    labels=['GS','HIO','HIO–ER','NCPP','Direct ResCNN','Direct E–D','SiSPRNet','FourierGSNet','Amplitude only','Shared 4-stage']
    fig,axs=plt.subplots(1,3,figsize=(5.25,3.5),gridspec_kw={'wspace':.34})
    xmin=1.;xmax=0.
    for i,(method,label) in enumerate(zip(methods,labels)):
        if method=='fouriergs_paired':ssim={'mean':modern['macro']['phase_foreground_ssim'],'ci95':modern['ci95']['phase_foreground_ssim']};rmse={'mean':modern['macro']['phase_foreground_rmse_rad'],'ci95':modern['ci95']['phase_foreground_rmse_rad']};ms=modern['latency_ms']['median']
        else:
            r=old[method];ssim={'mean':r['macro']['phase_foreground_ssim'],'ci95':r['ci95']['phase_foreground_ssim']};rmse={'mean':diag[method]['macro']['phase_foreground_rmse_rad'],'ci95':diag[method]['ci95']['phase_foreground_rmse_rad']}
            ms=r['latency_ms']['median']
        xmin=min(xmin,ssim['ci95'][0]);xmax=max(xmax,rmse['ci95'][1])
        point(axs[0],ssim,i,COLORS[0]);point(axs[1],rmse,i,COLORS[0]);axs[2].plot(ms,i,'s',color=COLORS[1],ms=3.5)
    for i,ax in enumerate(axs):
        clean(ax);ax.set_ylim(len(methods)-.6,-.6);ax.set_yticks(range(len(methods)),labels if i==0 else ['']*len(methods));ax.set_title('('+chr(97+i)+')',loc='left',fontweight='bold')
    axs[0].set_xlim(max(-1,math.floor((xmin-.025)*10)/10),1.01);axs[1].set_xlim(0,xmax*1.06)
    axs[2].set_xscale('log');axs[2].set_xlim(.8,10000)
    axs[0].set_xlabel('Foreground\nphase SSIM');axs[1].set_xlabel('Foreground phase\nRMSE (rad)');axs[2].set_xlabel('Reconstruction\ntime (ms)')
    fig.subplots_adjust(left=.24,right=.985,bottom=.16,top=.94);save(fig,'Fig_03_primary_comparison')

def stage_depth():
    old=json.loads((s.ROOT/'outputs/single_seed_experiments/summary.json').read_text(encoding='utf8'))['results']
    diag=json.loads((s.ROOT/'outputs/reviewer_revision_evidence/reviewer_diagnostics.json').read_text(encoding='utf8'))['results']
    stages=[1,2,3,4,6,8];fig,axs=plt.subplots(1,3,figsize=(5.25,2.45))
    for j in [0,1]:
        values=[];lower=[];upper=[]
        for n in stages:
            arm='stage'+str(n);r=old[arm] if j==0 else diag[arm];key='phase_foreground_ssim' if j==0 else 'phase_foreground_rmse_rad'
            values.append(r['macro'][key]);lower.append(r['ci95'][key][0]);upper.append(r['ci95'][key][1])
        vv=np.array(values);axs[j].errorbar(stages,vv,yerr=np.array([vv-np.array(lower),np.array(upper)-vv]),fmt='o-',color='#0072B2',ms=3,capsize=2,lw=.9,label='Common schedule')
        vals=[];lo=[];hi=[]
        for n in [6,8]:
            arm='stage'+str(n)+'_lr1e4';r=old[arm] if j==0 else diag[arm];key='phase_foreground_ssim' if j==0 else 'phase_foreground_rmse_rad'
            vals.append(r['macro'][key]);lo.append(r['ci95'][key][0]);hi.append(r['ci95'][key][1])
        vv=np.array(vals);axs[j].errorbar([6,8],vv,yerr=np.array([vv-np.array(lo),np.array(hi)-vv]),fmt='s--',color='#D55E00',ms=3,capsize=2,lw=.9,label='Lower rate')
        # Every conditional interval is visible.
        minimum=min(lower+lo);maximum=max(upper+hi);pad=max((maximum-minimum)*.12,.004)
        axs[j].set_ylim(max(0,minimum-pad),maximum+pad)
    axs[2].plot(stages,[old['stage'+str(n)]['latency_ms']['median'] for n in stages],'o-',color='#0072B2',ms=3)
    for i,ax in enumerate(axs):
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,lw=.5);ax.set_xlim(.6,8.4);ax.set_xticks(stages);ax.set_xlabel('Reconstruction stages');ax.set_title('('+chr(97+i)+')',loc='left',fontweight='bold')
    axs[0].set_ylabel('Foreground phase SSIM');axs[1].set_ylabel('Foreground RMSE (rad)');axs[2].set_ylabel('Median inference time (ms)')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.11,right=.985,bottom=.28,top=.9,wspace=.55);save(fig,'Fig_05_stage_count')

def conditions(a):
    conditions=v.CONDITIONS[1:];labels=['Coupled','Different-source absorption','Pure phase']
    groups=[['stage4','calibrated_unet','calibrated_sisprnet','fouriergs_paired','HIO','NCPP'],['stage4_coupled','stage4_mixed','unet_coupled','unet_mixed']]
    names=[['Original 4-stage','Original E–D','SiSPRNet','FourierGSNet','HIO','NCPP'],['4-stage coupled','4-stage mixed','E–D coupled','E–D mixed']]
    fig,axs=plt.subplots(1,2,figsize=(5.25,3.3),gridspec_kw={'wspace':.62})
    maximum=max(a['results'][c][m]['phase_foreground_rmse_rad']['ci95'][1] for methods in groups for m in methods for c in conditions)
    for p,(ax,methods,display) in enumerate(zip(axs,groups,names)):
        for j,condition in enumerate(conditions):
            for i,method in enumerate(methods):point(ax,a['results'][condition][method]['phase_foreground_rmse_rad'],i+(j-1)*.19,COLORS[j],['o','s','^'][j])
        clean(ax);ax.set_yticks(range(len(methods)),display);ax.set_ylim(len(methods)-.6,-.6);ax.set_xlim(0,max(1.12,maximum*1.06));ax.set_xlabel('Foreground phase RMSE (rad)')
        ax.set_title('('+chr(97+p)+') '+['Original training','Additional training'][p],loc='left',fontweight='bold')
    handles=[plt.Line2D([],[],marker=m,color=c,ls='',ms=4,label=l) for m,c,l in zip(['o','s','^'],COLORS,labels)]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8,columnspacing=1.1,handletextpad=.4)
    fig.subplots_adjust(left=.20,right=.985,bottom=.24,top=.92);save(fig,'Fig_07_object_conditions')

def qualitative():
    conditions=v.CONDITIONS[1:]
    balanced=s.ROOT/'work/balanced_pairing_v1'
    caches=[torch.load((balanced if c=='independent_absorption' else v.OUT)/(c+'_cache.pt'),weights_only=False) for c in conditions]
    # Source choice is fixed independently of reconstruction quality.
    records=caches[0]['records'];ix=min(range(len(records)),key=lambda i:(records[i]['cell_type'],records[i]['source_qpm'],records[i]['simulation_index']))
    source=records[ix];assert all(c['records'][ix]['content_id']==source['content_id'] for c in caches)
    fig,axs=plt.subplots(3,6,figsize=(5.25,3.40));titles=['Reference\nphase','Measured\nabsorption','Screen\nmodulus','Original\n4-stage','Mixed\n4-stage','Mixed\nabsolute error']
    mu=[]
    for cache in caches:
        b=[t[ix:ix+1] for t in cache['data']];pp=b[2]
        calibrated=(b[0]*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/s._camera_modulus_gain(pp)
        scale=(pp[:,0:1]*pp[:,1:2]/pp[:,2:3].square())[:,:,None,None]*128**2
        mu.append((torch.log1p(100*calibrated/scale)/math.log(101))[0,0].numpy())
    mu_max=max(float(x.max()) for x in mu)
    for row,(condition,cache) in enumerate(zip(conditions,caches)):
        data=cache['data'];truth=data[7][ix,1].numpy()
        absorption=(1-data[3][ix,0]/data[4][ix,0].clamp_min(1e-4)).numpy()
        displaymask=(data[4][ix,0]>=.12*data[4][ix,0].max()).numpy()
        absorption=np.ma.array(absorption,mask=~displaymask)
        originalpath=balanced/'stage4/predictions.pt' if condition=='independent_absorption' else v.OUT/condition/'stage4/predictions.pt'
        mixedpath=balanced/'stage4_mixed/predictions.pt' if condition=='independent_absorption' else s.ROOT/'work/reviewer_training_v1/evaluation'/condition/'stage4_mixed/predictions.pt'
        original=torch.load(originalpath,weights_only=False)['predictions'][ix,1].numpy()
        mixed=torch.load(mixedpath,weights_only=False)['predictions'][ix,1].numpy()
        fields=[truth,absorption,mu[row],original,mixed,np.abs(mixed-truth)]
        for col,(ax,field) in enumerate(zip(axs[row],fields)):
            if col in [0,3,4]:im=ax.imshow(field,cmap='viridis',vmin=0,vmax=math.pi)
            elif col==1:im=ax.imshow(field,cmap='coolwarm',vmin=-.05,vmax=.1)
            elif col==2:im=ax.imshow(field,cmap='gray',vmin=0,vmax=mu_max)
            else:im=ax.imshow(field,cmap='magma',vmin=0,vmax=1)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=8)
            if col==0:
                ax.set_ylabel(['Coupled','Different-source\nabsorption','Pure phase'][row],fontsize=8)
                pp=data[2][ix];sampling=float(pp[0]*pp[1]/pp[2]);pixels=50e-6/sampling
                ax.plot([6,6+pixels],[118,118],color='white',lw=1.4)
            if row==2:
                cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.04)
                cb.ax.tick_params(labelsize=7,length=2)
                if col in [0,3,4]:cb.set_ticks([0,math.pi]);cb.set_ticklabels(['0','π'])
                elif col==1:cb.set_ticks([-.05,.1]);cb.set_ticklabels(['−0.05','0.10'])
                elif col==2:cb.set_ticks([0,mu_max]);cb.set_ticklabels(['0',f'{mu_max:.2f}'])
                elif col==5:cb.set_ticks([0,1]);cb.set_ticklabels(['0','≥1'])
                labels=cb.ax.get_xticklabels()
                labels[0].set_ha('left');labels[-1].set_ha('right')
    fig.subplots_adjust(left=.1,right=.995,bottom=.12,top=.86,wspace=.08,hspace=.08)
    save(fig,'Fig_08_object_condition_examples')
    phase_source={k:source[k] for k in ['source_qpm','content_id','cell_type','simulation_index']}
    records_by_condition={c:{**cache['records'][ix],'absorption_donor_used':c=='independent_absorption'} for c,cache in zip(conditions,caches)}
    s.dump(OUT/'qualitative_selection.json',{'rule':'first category, source path and optical realization; no error selection','phase_source':phase_source,'records_by_condition':records_by_condition,'screen_display':'log(1+100*calibrated_modulus/FFTscale)/log101','screen_display_max':mu_max,'phase_limits':[0,math.pi],'error_limits_rad':[0,1],'error_colors_saturate_above_1rad':True,'absorption_display_limits':[-.05,.1],'absorption_display_mask':'observed flat >=12% image maximum; display only','scale_bar_um':50})

def learning():
    fig,axs=plt.subplots(1,3,figsize=(5.25,2.3),sharey=True)
    arms=['stage4_mixed','stage4_coupled','unet_mixed','unet_coupled'];styles=[('#0072B2','-'),('#0072B2','--'),('#D55E00','-'),('#D55E00','--')]
    for ax,c,title in zip(axs,v.CONDITIONS[1:],['Coupled','Different-source\nabsorption','Pure phase']):
        for arm,(color,ls) in zip(arms,styles):
            hist=json.loads((s.ROOT/'work/reviewer_training_v1'/arm/'history.json').read_text())
            ax.plot([r['epoch'] for r in hist],[r['validation_conditions'][c]['phase_foreground_ssim'] for r in hist],color=color,ls=ls,label=arm)
            selected=json.loads((s.ROOT/'work/reviewer_training_v1'/arm/'complete.json').read_text(encoding='utf8'))['selected_epoch']
            ax.plot(selected,hist[selected-1]['validation_conditions'][c]['phase_foreground_ssim'],'o',color=color,ms=2.6)
        ax.set_title(title,fontsize=7.5);ax.set_xlabel('Additional epoch');ax.set_xlim(1,40);ax.set_ylim(0,1);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,lw=.5)
    axs[0].set_ylabel('Validation foreground SSIM')
    names=['4-stage mixed','4-stage coupled','E–D mixed','E–D coupled'];handles=[plt.Line2D([],[],color=c,ls=ls,label=l) for (c,ls),l in zip(styles,names)]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=7.3)
    fig.subplots_adjust(left=.1,right=.99,top=.88,bottom=.35,wspace=.2);save(fig,'Fig_S10_corrective_learning')

def new_control_learning(modern):
    nodiff=json.loads((s.ROOT/'work/no_diffraction_v1/internal_summary.json').read_text(encoding='utf8'))
    arms=[(s.OUT/'learning/stage4','Shared four stages','#0072B2',json.loads((s.OUT/'learning/stage4/complete.json').read_text(encoding='utf8'))['best_epoch']),
      (s.ROOT/'work/no_diffraction_v1/learning/stage4_no_diffraction','Without diffraction','#666666',nodiff['selected_epoch']),
      (s.ROOT/'work/fouriergs_v1/learning/fouriergs_paired','FourierGSNet','#D55E00',modern['selected_epoch'])]
    fig,axs=plt.subplots(1,2,figsize=(5.25,2.5))
    for folder,label,color,selected in arms:
        history=json.loads((folder/'history.json').read_text(encoding='utf8'))
        assert len(history)==40
        for ax,key in zip(axs,['phase_foreground_ssim','selection_score']):
            values=[r['selection_score'] if key=='selection_score' else r['validation_macro'][key] for r in history]
            ax.plot([r['epoch'] for r in history],values,label=label,color=color)
            ax.plot(selected,values[selected-1],'o',color=color,ms=3)
    for i,ax in enumerate(axs):
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15,lw=.5);ax.set_xlim(1,40);ax.set_xlabel('Training epoch');ax.set_title('('+chr(97+i)+')',loc='left',fontweight='bold')
    axs[0].set_ylabel('Validation foreground SSIM');axs[0].set_ylim(0,1)
    axs[1].set_ylabel('Validation selection score')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8,columnspacing=1.)
    fig.subplots_adjust(left=.1,right=.98,bottom=.27,top=.9,wspace=.4);save(fig,'Fig_S11_new_control_learning')

def main():
    assert (s.ROOT/'work/fouriergs_v1/all_complete.json').exists()
    a=json.loads((s.ROOT/'work/evidence_analysis/evidence_analysis.json').read_text());m=json.loads((s.ROOT/'work/fouriergs_v1/internal_summary.json').read_text())
    comparison(a,m);stage_depth();conditions(a);qualitative();learning();new_control_learning(m)
    print('Created six 600dpi scientific raster figures.',flush=True)

if __name__=='__main__':main()
