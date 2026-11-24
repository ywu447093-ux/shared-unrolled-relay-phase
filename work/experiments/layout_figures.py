"""Replot saved scientific data for legible final-width figures; no inference."""
from pathlib import Path
import json,math
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import suite as s

OUT=s.ROOT/'work/layout_revision/figures';OUT.mkdir(parents=True,exist_ok=True)
OLD=json.loads((s.ROOT/'outputs/single_seed_experiments/summary.json').read_text())['results']
D=json.loads((s.ROOT/'outputs/reviewer_revision_evidence/reviewer_diagnostics.json').read_text())['results']
F=json.loads((s.ROOT/'work/fouriergs_v1/internal_summary.json').read_text())
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8.5,
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold',
 'axes.linewidth':.65,'lines.linewidth':1.,'figure.facecolor':'white','savefig.dpi':600})
BLUE='#0072B2';ORANGE='#D55E00';BLACK='#303030'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white');plt.close(fig)
def clean(ax,axis='x'):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis=axis,color='#e4e4e4',linewidth=.45);ax.set_axisbelow(True)

def comparison():
    methods=['GS','HIO','HIO_ER','NCPP','calibrated_rescnn','calibrated_unet','calibrated_sisprnet','fouriergs_paired','amplitude_only','stage4']
    labels=['GS','HIO','HIO–ER','NCPP','Residual CNN','Encoder–decoder','SiSPRNet','FourierGSNet','Amplitude only','Shared four stages']
    fig,axs=plt.subplots(1,2,figsize=(5.25,2.68),sharey=True)
    records=[]
    for i,method in enumerate(methods):
        record={'method':method}
        color,marker=(BLACK,'o') if i<4 else (BLUE,'s') if i<9 else (ORANGE,'D')
        for ax,key in zip(axs,['phase_foreground_rmse_rad','phase_foreground_ssim']):
            r=F if method=='fouriergs_paired' else (D[method] if key.endswith('rad') else OLD[method])
            mu=r['macro'][key];lo,hi=r['ci95'][key];record[key]={'mean':mu,'ci95':[lo,hi]}
            ax.errorbar(mu,i,xerr=[[mu-lo],[hi-mu]],fmt=marker,color=color,mfc='white' if i<4 else color,ms=3.4,capsize=2,lw=.8)
        records.append(record)
    for i,ax in enumerate(axs):
        clean(ax);ax.axhspan(8.55,9.45,color='#f4f4f4',zorder=-1);ax.set_ylim(9.6,-.6)
        ax.set_yticks(range(10));ax.tick_params(axis='y',length=0,pad=5)
        ax.set_title(['(a) Absolute phase error','(b) Structural fidelity'][i],loc='left',pad=7)
    axs[0].set_yticklabels(labels);axs[1].tick_params(labelleft=False)
    axs[0].set_xlim(0,.38);axs[0].set_xticks([0,.1,.2,.3]);axs[0].set_xlabel('Foreground RMSE (rad)')
    axs[1].set_xlim(.4,1.01);axs[1].set_xticks([.4,.6,.8,1.]);axs[1].set_xlabel('Foreground SSIM')
    fig.subplots_adjust(left=.245,right=.985,top=.87,bottom=.19,wspace=.29);save(fig,'Fig_03_primary_comparison')
    return records

def depth():
    stages=[1,2,3,4,6,8];fig,axs=plt.subplots(1,2,figsize=(5.25,2.48))
    records={}
    for j,(ax,key) in enumerate(zip(axs,['phase_foreground_rmse_rad','phase_foreground_ssim'])):
        for arms,xs,color,fmt,label in [(['stage'+str(n) for n in stages],stages,BLUE,'o-','Common schedule'),(['stage6_lr1e4','stage8_lr1e4'],[6,8],ORANGE,'s--','Lower rate, 6 and 8 stages')]:
            rr=[(D[a] if j==0 else OLD[a]) for a in arms]
            mu=np.array([r['macro'][key] for r in rr]);lo=np.array([r['ci95'][key][0] for r in rr]);hi=np.array([r['ci95'][key][1] for r in rr])
            ax.errorbar(xs,mu,yerr=[mu-lo,hi-mu],fmt=fmt,color=color,ms=3.5,capsize=2,lw=1.,label=label)
            for arm,m,l,h in zip(arms,mu,lo,hi):records.setdefault(arm,{})[key]={'mean':float(m),'ci95':[float(l),float(h)]}
        clean(ax,'y');ax.axvspan(2.75,4.25,color='#f2f2f2',zorder=-1);ax.set_xlim(.65,8.35);ax.set_xticks(stages);ax.set_xlabel('Reconstruction stages')
        ax.set_title(['(a) Absolute phase error','(b) Structural fidelity'][j],loc='left',pad=7)
    axs[0].set_ylabel('Foreground RMSE (rad)');axs[0].set_ylim(.045,.16);axs[0].set_yticks([.05,.075,.10,.125,.15])
    axs[1].set_ylabel('Foreground SSIM');axs[1].set_ylim(.865,.97);axs[1].set_yticks([.88,.90,.92,.94,.96])
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8,columnspacing=1.1)
    fig.subplots_adjust(left=.11,right=.985,top=.88,bottom=.28,wspace=.4);save(fig,'Fig_05_stage_count')
    return records

def mixing_sharing():
    groups=[['no_global','no_attention','heads1','stage4','heads8'],['stage4','private_heads','private_decoders','unshared']]
    names=[['CNN bottleneck','FFN only','One head','Four heads','Eight heads'],['Shared','Private heads','Private decoders','Independent']]
    fig,axs=plt.subplots(1,2,figsize=(5.25,2.35));records={}
    for j,(ax,methods,labels) in enumerate(zip(axs,groups,names)):
        for i,m in enumerate(methods):
            r=OLD[m];key='phase_foreground_ssim';mu=r['macro'][key];lo,hi=r['ci95'][key]
            records[m]={'mean':mu,'ci95':[lo,hi]}
            color=ORANGE if m=='stage4' else BLUE
            ax.errorbar(mu,i,xerr=[[mu-lo],[hi-mu]],fmt='D' if m=='stage4' else 'o',color=color,ms=3.4,capsize=2,lw=.8)
        clean(ax);ax.set_yticks(range(len(methods)),labels);ax.set_ylim(len(methods)-.5,-.5)
        ax.set_xlabel('Foreground phase SSIM');ax.set_title(['(a) Global mixing','(b) Weight sharing'][j],loc='left',pad=7)
        ax.tick_params(axis='y',length=0,pad=4)
    for ax in axs:ax.set_xlim(.93,.96);ax.set_xticks([.93,.94,.95,.96])
    fig.subplots_adjust(left=.21,right=.96,top=.85,bottom=.23,wspace=.90)
    save(fig,'Fig_06_mixing_sharing');return records

def plates():
    example=torch.load(s.ROOT/'outputs/single_seed_experiments/example_inputs.pt',weights_only=False)
    names=example['names'];pred={}
    methods=['HIO_ER','GS','NCPP','calibrated_unet','calibrated_sisprnet','stage4','fouriergs_paired']
    for m in methods:
        folder=(s.ROOT/'work/fouriergs_v1/learning'/m) if m=='fouriergs_paired' else s.OUT/('classical' if m in ['HIO_ER','GS','NCPP'] else 'learning')/m
        data=torch.load(folder/'predictions.pt',weights_only=False);ix={n:i for i,n in enumerate(data['names'])};pred[m]=data['predictions'][[ix[n] for n in names],1].numpy()
    truth=example['references'][:,1].numpy()
    for keys,titles,fname in [(['reference','HIO_ER','calibrated_unet','stage4','error'],['Reference','HIO–ER','Direct E–D','Shared four stages','Absolute error'],'Fig_04_reconstructions'),
       (['reference','GS','NCPP','calibrated_sisprnet','fouriergs_paired','stage4'],['Reference','GS','NCPP','SiSPRNet','FourierGSNet','Shared four stages'],'Fig_S12_additional_reconstructions')]:
        ncol=len(keys);fig,axs=plt.subplots(5,ncol,figsize=(5.25,4.72 if ncol==5 else 4.08))
        for row in range(5):
            for col,(key,ax) in enumerate(zip(keys,axs[row])):
                img=truth[row] if key=='reference' else abs(pred['stage4'][row]-truth[row]) if key=='error' else pred[key][row]
                ax.imshow(img,cmap='magma',vmin=0,vmax=.65 if key=='error' else math.pi);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:ax.set_title(titles[col].replace('Shared four stages','Shared\nfour stages'),fontsize=8 if ncol==5 else 7.7,pad=5)
                if col==0:
                    ax.set_ylabel(example['categories'][row],fontsize=8.5)
                    pp=example['observations'][2][row];sampling=float(pp[0]*pp[1]/pp[2]);ax.plot([6,6+50e-6/sampling],[118,118],color='white',lw=1.5)
        fig.subplots_adjust(left=.09,right=.99,top=.92,bottom=.15,wspace=.07,hspace=.08)
        end=.78 if 'error' in keys else .98;cax=fig.add_axes([.12,.075,end-.12,.012]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,math.pi),cmap='magma'),cax=cax,orientation='horizontal');cb.set_ticks([0,math.pi/2,math.pi],labels=['0','π/2','π']);cb.ax.tick_params(labelsize=8,length=2);fig.text(.12,.107,'Phase (rad)',fontsize=8)
        if 'error' in keys:
            cax=fig.add_axes([.82,.075,.16,.012]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,.65),cmap='magma'),cax=cax,orientation='horizontal');cb.set_ticks([0,.65],labels=['0','0.65']);cb.ax.tick_params(labelsize=8,length=2);fig.text(.82,.107,'Error (rad)',fontsize=8)
        save(fig,fname)
    return {'names':names,'categories':example['categories'],'selection':example['selection'],'phase_range_rad':[0,math.pi],'error_range_rad':[0,.65],'scale_bar_um':50}

if __name__=='__main__':
    records={'comparison':comparison(),'depth':depth(),'mixing_sharing':mixing_sharing(),'plates':plates(),'no_new_predictions':True}
    (OUT/'figure_data.json').write_text(json.dumps(records,indent=2),encoding='utf8');print('Created five figures from unchanged predictions and statistical records.')
