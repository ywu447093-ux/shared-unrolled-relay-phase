"""Final-width scientific plots from frozen numerical records and saved pixels."""
from pathlib import Path
from types import SimpleNamespace
import ast,csv,json,math,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
OUT=ROOT/'outputs/JOSA_A_Refined_Figures';OUT.mkdir(exist_ok=True)
for font in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_manager.fontManager.addfont('C:/Windows/Fonts/'+font)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.linewidth':.65,'lines.linewidth':1.,'figure.facecolor':'white','savefig.dpi':600,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold','errorbar.capsize':2.5})
BLUE='#0072B2';ORANGE='#D55E00';BLACK='#303030';COLORS=[BLUE,ORANGE,'#009E73']
OLD=json.loads((ROOT/'outputs/single_seed_experiments/summary.json').read_text())['results']
D=json.loads((ROOT/'outputs/reviewer_revision_evidence/reviewer_diagnostics.json').read_text())['results']
F=json.loads((ROOT/'work/fouriergs_v1/internal_summary.json').read_text())
A=json.loads((ROOT/'work/evidence_analysis/evidence_analysis.json').read_text())
Z=np.load(W/'saved_arrays.npz');meta=json.loads((W/'saved_arrays_metadata.json').read_text())
truth=Z['truth'];categories=meta['categories'];C=ROOT/'work/experiment_results_v1'
s=SimpleNamespace(ROOT=ROOT,OUT=C);v=SimpleNamespace(CONDITIONS=['secondary','fresh_coupled','independent_absorption','pure_phase'])
receipt={};data_log={};SINGLE=False
def clean(ax,axis='x'):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis=axis,color='#e5e5e5',linewidth=.45);ax.set_axisbelow(True)
def grid(ax):clean(ax,'both')
def save(fig,name):
    if name=='Fig_S10_corrective_learning':fig.subplots_adjust(top=.81,right=.975)
    if name=='Fig_07_object_conditions' and not SINGLE:fig.set_size_inches(5.25,3.0)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];fonts=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        # Hidden ticks beyond the view can be retained by Matplotlib.
        if b.x1<0 or b.x0>fig.bbox.width or b.y1<0 or b.y0>fig.bbox.height:continue
        if b.x0<-.6 or b.y0<-.6 or b.x1>fig.bbox.width+.6 or b.y1>fig.bbox.height+.6:outside.append(t.get_text())
        fonts.append(t.get_fontsize())
    assert not outside,(name,'text outside figure',outside)
    non_image=[ax for ax in fig.axes if not ax.images and ax.get_xlabel()]
    if not SINGLE:
        for ax in non_image:assert not ax.spines['top'].get_visible() and not ax.spines['right'].get_visible(),name
    folder=OUT/'single_column' if SINGLE else OUT;folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=600,facecolor='white')
    errors=[]
    for ax in fig.axes:
        for c in ax.containers:
            if not isinstance(c,matplotlib.container.ErrorbarContainer):continue
            for cap in c.lines[1]:assert abs(cap.get_markersize()-5)<1e-8,(name,'inconsistent capsize')
            for lines in c.lines[2]:
                for segment in lines.get_segments():
                    if not len(segment):continue
                    xlim=sorted(ax.get_xlim());ylim=sorted(ax.get_ylim())
                    assert np.min(segment[:,0])>=xlim[0]-1e-8 and np.max(segment[:,0])<=xlim[1]+1e-8,(name,'CI clipped in x')
                    assert np.min(segment[:,1])>=ylim[0]-1e-8 and np.max(segment[:,1])<=ylim[1]+1e-8,(name,'CI clipped in y')
                    errors.append(segment.tolist())
    legends=[t.get_text() for leg in list(fig.legends)+[ax.get_legend() for ax in fig.axes if ax.get_legend()] for t in leg.get_texts()]
    receipt[('single_column/' if SINGLE else '')+name]={'width_in':fig.get_figwidth(),'height_in':fig.get_figheight(),'minimum_text_pt':min(fonts) if fonts else None,'font':'Times New Roman','capsize_pt':2.5 if errors else None,'errorbars':'95% percentile bootstrap CI from 2000 category-stratified source-content resamples, conditional on fixed models' if errors else None,'errorbar_segments':errors,'legend_labels':legends,'curve_styles':sorted(set(line.get_linestyle() for ax in fig.axes for line in ax.lines)),'text_inside_canvas':True,'confidence_intervals_inside_axes':True,'data_unchanged':True}
    plt.close(fig)

# Reuse pure plotting functions, never import or execute training modules.
source=ast.parse((ROOT/'work/experiments/layout_figures.py').read_text())
selected=[x for x in source.body if isinstance(x,ast.FunctionDef) and x.name in ['comparison','depth','mixing_sharing']]
txt=ast.unparse(ast.Module(body=selected,type_ignores=[])).replace('capsize=2','capsize=2.5')
exec(compile(txt,'saved-layout-functions','exec'),globals())
old_save=save
def save(fig,name):
    # Add explicit legends without changing any plotted values.
    if name=='Fig_03_primary_comparison':
        fig.set_size_inches(5.25,2.85);fig.subplots_adjust(top=.80,bottom=.17)
        handles=[plt.Line2D([],[],marker=m,ls='',color=c,mfc=f,label=l,ms=3.5) for m,c,f,l in [('o',BLACK,'white','Classical'),('s',BLUE,BLUE,'Learned baseline'),('D',ORANGE,ORANGE,'Shared four-stage')]]
        fig.legend(handles=handles,ncol=3,loc='upper center',frameon=False,bbox_to_anchor=(.57,1.0),columnspacing=1.1,handletextpad=.35)
    elif name=='Fig_05_stage_count':
        for leg in fig.legends:leg.remove()
        hh,ll=fig.axes[0].get_legend_handles_labels()
        fig.legend(hh,['Common rate (3 × 10⁻⁴)','Lower rate (10⁻⁴)'],loc='lower center',ncol=2,frameon=False,fontsize=8.5,columnspacing=1.1)
    elif name=='Fig_06_mixing_sharing':
        fig.set_size_inches(5.25,2.45);fig.subplots_adjust(bottom=.30,top=.86,wspace=.82)
        hh=[plt.Line2D([],[],marker=m,ls='',color=c,label=l,ms=3.5) for m,c,l in [('o',BLUE,'Ablation variant'),('D',ORANGE,'Shared reference')]]
        fig.legend(handles=hh,loc='lower center',ncol=2,frameon=False)
    old_save(fig,name)
data_log['comparison']=comparison();data_log['depth']=depth();data_log['mixing_sharing']=mixing_sharing()
save=old_save

def plates():
    for keys,titles,name in [(['reference','HIO_ER','calibrated_unet','stage4','error'],['Reference','HIO–ER','Direct E–D','Shared\nfour stages','Absolute error'],'Fig_04_reconstructions'),(['reference','GS','NCPP','calibrated_sisprnet','fouriergs_paired','stage4'],['Reference','GS','NCPP','SiSPRNet','FourierGSNet','Shared\nfour stages'],'Fig_S12_additional_reconstructions')]:
        n=len(keys);fig,axs=plt.subplots(5,n,figsize=(5.25,4.72 if n==5 else 4.08))
        for i in range(5):
            for j,(key,ax) in enumerate(zip(keys,axs[i])):
                image=truth[i] if key=='reference' else abs(Z['pred_stage4'][i]-truth[i]) if key=='error' else Z['pred_'+key][i]
                ax.imshow(image,cmap='magma',vmin=0,vmax=.65 if key=='error' else math.pi);ax.set(xticks=[],yticks=[])
                for sp in ax.spines.values():sp.set_visible(False)
                if i==0:ax.set_title(titles[j],fontsize=8.5,pad=5)
                if j==0:
                    ax.set_ylabel(categories[i]);pp=Z['geometry'][i];sampling=float(pp[0]*pp[1]/pp[2]);ax.plot([6,6+50e-6/sampling],[118,118],color='white',lw=1.5)
        fig.subplots_adjust(left=.09,right=.99,top=.92,bottom=.15,wspace=.07,hspace=.08)
        end=.78 if 'error' in keys else .98;cax=fig.add_axes([.12,.075,end-.12,.012]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,math.pi),cmap='magma'),cax=cax,orientation='horizontal');cb.set_ticks([0,math.pi/2,math.pi],labels=['0','π/2','π']);cb.ax.tick_params(length=2);fig.text(.12,.107,'Phase (rad)')
        if 'error' in keys:
            cax=fig.add_axes([.82,.075,.16,.012]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,.65),cmap='magma'),cax=cax,orientation='horizontal');cb.set_ticks([0,.65],labels=['0','0.65']);cb.ax.tick_params(length=2);fig.text(.82,.107,'Error (rad)')
        save(fig,name)
plates()

src=ast.parse((ROOT/'work/experiments/evidence_figures.py').read_text())
funcs=[x for x in src.body if isinstance(x,ast.FunctionDef) and x.name in ['point','conditions','learning','new_control_learning']]
text=re.sub(r'fontsize=(?:7\.5|7\.3|8)(?![\d.])','fontsize=8.5',ast.unparse(ast.Module(body=funcs,type_ignores=[])).replace('capsize=2','capsize=2.5'))
# Distinct line styles provide method identification in monochrome.
text=text.replace("ax.plot([r['epoch'] for r in history], values, label=label, color=color)","ax.plot([r['epoch'] for r in history], values, label=label, color=color, ls={'Shared four stages':'-','Without diffraction':'--','FourierGSNet':'-.'}[label])")
exec(compile(text,'saved-evidence-functions','exec'),globals())
conditions(A);learning();new_control_learning(F)

def qualitative():
    fig,axs=plt.subplots(3,6,figsize=(5.25,3.4));titles=['Reference\nphase','Measured\nabsorption','Screen\nmodulus','Original\n4-stage','Mixed\n4-stage','Mixed\nabsolute error'];umax=max(Z[f'condition_{i}_screen'].max() for i in range(3))
    for row in range(3):
        ref=Z[f'condition_{row}_truth'];mixed=Z[f'condition_{row}_mixed']
        fields=[ref,np.ma.array(Z[f'condition_{row}_absorption'],mask=~Z[f'condition_{row}_mask']),Z[f'condition_{row}_screen'],Z[f'condition_{row}_original'],mixed,abs(mixed-ref)]
        for col,(ax,field) in enumerate(zip(axs[row],fields)):
            cm,lo,hi=('viridis',0,math.pi) if col in [0,3,4] else ('coolwarm',-.05,.1) if col==1 else ('gray',0,umax) if col==2 else ('magma',0,1)
            im=ax.imshow(field,cmap=cm,vmin=lo,vmax=hi);ax.set(xticks=[],yticks=[])
            for sp in ax.spines.values():sp.set_visible(False)
            if row==0:ax.set_title(titles[col],fontsize=8.5)
            if col==0:
                ax.set_ylabel(['Coupled','Different-source\nabsorption','Pure phase'][row],fontsize=8.5)
                pp=Z[f'condition_{row}_geometry'];ax.plot([6,6+50e-6/float(pp[0]*pp[1]/pp[2])],[118,118],color='white',lw=1.4)
            if row==2:
                cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.04);cb.ax.tick_params(labelsize=8,length=2)
                ticks=[0,math.pi] if col in [0,3,4] else [-.05,.1] if col==1 else [0,umax] if col==2 else [0,1]
                labels=['0','π'] if col in [0,3,4] else ['−0.05','0.10'] if col==1 else ['0',f'{umax:.2f}'] if col==2 else ['0','≥1']
                cb.set_ticks(ticks,labels=labels);tt=cb.ax.get_xticklabels();tt[0].set_ha('left');tt[-1].set_ha('right')
    fig.subplots_adjust(left=.1,right=.995,bottom=.12,top=.86,wspace=.08,hspace=.08);save(fig,'Fig_08_object_condition_examples')
qualitative()

# Other supplementary figures retain their source pixels, profile rows and histories.
fig,axs=plt.subplots(1,5,figsize=(5.25,1.6))
arrays=[truth[0],Z['amplitude'][0],np.log1p(100*Z['obs_sps'][0,0])/math.log(101),Z['obs_amp'][0,0],Z['obs_support'][0,0]]
for ax,arr,title,maxi in zip(axs,arrays,['(a) Phase','(b) Clean A','(c) SPS log','(d) Noisy A','(e) Support'],[math.pi,1.25,1,1.25,1]):
    im=ax.imshow(arr,cmap='magma' if title=='(a) Phase' else 'gray',vmin=0,vmax=maxi);ax.set(xticks=[],yticks=[]);ax.set_title(title)
    for sp in ax.spines.values():sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.07);cb.set_ticks([0,maxi],labels=['0','π' if maxi==math.pi else str(maxi).rstrip('0').rstrip('.')]);cb.ax.tick_params(labelsize=8)
    tt=cb.ax.get_xticklabels();tt[0].set_ha('left');tt[-1].set_ha('right')
fig.subplots_adjust(left=.03,right=.98,top=.82,bottom=.20,wspace=.28);save(fig,'Fig_S01_simulated_channels')

stages=['stage1','stage2','stage3','stage4','stage6','stage8'];mix=['no_global','no_attention','heads1','stage4','heads8'];share=['stage4','private_heads','private_decoders','unshared']
groups=[stages+['stage6_lr1e4','stage8_lr1e4'],mix,share,['stage4','calibrated_unet','calibrated_rescnn','calibrated_sisprnet','amplitude_only']]
short={'stage1':'1 stage','stage2':'2 stages','stage3':'3 stages','stage4':'4 stages','stage6':'6 stages','stage8':'8 stages','stage6_lr1e4':'6, low rate','stage8_lr1e4':'8, low rate','no_global':'CNN bottleneck','no_attention':'FFN only','heads1':'1 head','heads8':'8 heads','private_heads':'Private heads','private_decoders':'Private decoders','unshared':'Independent','calibrated_unet':'Encoder–decoder','calibrated_rescnn':'Residual CNN','calibrated_sisprnet':'SiSPRNet','amplitude_only':'Amplitude only'}
short.update({'calibrated_unet':'E–D','calibrated_rescnn':'ResCNN','amplitude_only':'Amplitude'})
palette=[BLUE,ORANGE,'#009E73','#CC79A7','#E69F00','#000000','#56B4E9','#777777']
fig,axs=plt.subplots(2,2,figsize=(5.25,5.5))
for panel,(ax,arms,title) in enumerate(zip(axs.flat,groups,['Depth','Global mixing','Weight sharing','Direct controls'])):
    for j,arm in enumerate(arms):
        hist=json.loads((C/'learning'/arm/'history.json').read_text());ax.plot([r['epoch'] for r in hist],[r['validation_macro']['phase_foreground_ssim'] for r in hist],color=palette[j],ls=['-','--',':','-.'][j%4],marker=['o','s','^','D'][j%4],markevery=8,ms=2,label=short[arm])
    ax.set(xlabel='Training epoch',ylabel='Validation phase SSIM',title=f'({chr(97+panel)}) {title}',xticks=[0,20,40]);grid(ax)
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.28),handlelength=1.7,columnspacing=.65,fontsize=8.5)
fig.subplots_adjust(left=.125,right=.98,top=.95,bottom=.23,hspace=1.1,wspace=.42);save(fig,'Fig_S06_learning_curves')

curves=json.loads((C/'classical_validation_budget_curves.json').read_text());fig,axs=plt.subplots(1,2,figsize=(5.25,2.7))
for j,(method,rr) in enumerate(curves.items()):
    for ax,key in zip(axs,['foreground_ssim','selection_score']):ax.plot([r['propagation_pairs'] for r in rr],[r[key] for r in rr],ls=['-','--',':','-.'][j%4],marker=['o','s','^','D'][j%4],color=palette[j],ms=3,label=method.replace('_','–'))
for j,ax in enumerate(axs):
    ax.set(xscale='log',xlim=(80,12000),xticks=[100,1000,10000],xlabel='Propagation pairs',ylabel='Validation phase SSIM' if j==0 else 'Validation selection score');grid(ax);ax.set_title('('+chr(97+j)+')',loc='left')
hh,ll=axs[0].get_legend_handles_labels();fig.legend(hh,ll,ncol=4,loc='upper center',frameon=False,handlelength=1.7,columnspacing=1)
fig.subplots_adjust(left=.12,right=.98,top=.80,bottom=.20,wspace=.47);save(fig,'Fig_S07_classical_budget')

fig,axs=plt.subplots(1,5,figsize=(5.25,1.85));fig.subplots_adjust(left=.03,right=.98,top=.75,bottom=.28,wspace=.19)
arrays=[Z['worst_truth'],Z['worst_stage4'],Z['worst_calibrated_unet'],Z['worst_amplitude_only'],abs(Z['worst_stage4']-Z['worst_truth'])]
for j,(ax,arr,title) in enumerate(zip(axs,arrays,['Reference','Shared\n4-stage','Direct\nE–D','Amplitude\nonly','Absolute\nerror'])):
    ax.imshow(arr,cmap='magma',vmin=0,vmax=1 if j==4 else math.pi);ax.set(xticks=[],yticks=[]);ax.set_title(title)
    for sp in ax.spines.values():sp.set_visible(False)
for j in [0,4]:
    pos=axs[j].get_position();cx=fig.add_axes([pos.x0,.18,pos.width,.025]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1 if j==4 else math.pi),cmap='magma'),cax=cx,orientation='horizontal',extend='max' if j==4 and arrays[j].max()>1 else 'neither');cb.set_ticks([0,1 if j==4 else math.pi],labels=['0','1' if j==4 else 'π']);cb.set_label('rad',labelpad=0)
    tt=cb.ax.get_xticklabels();tt[0].set_ha('left');tt[-1].set_ha('right')
save(fig,'Fig_S08_difficult_case')

with (ROOT/'outputs/reviewer_revision_evidence/fixed_phase_profiles.csv').open(encoding='utf-8-sig',newline='') as fp:rows=list(csv.DictReader(fp))
line_methods=['reference_rad','HIO_ER','calibrated_unet','amplitude_only','stage4'];line_labels=['Reference','HIO–ER','Direct E–D','Amplitude only','Shared 4-stage'];colors=['#000000','#009E73','#E69F00','#CC79A7',BLUE];styles=['-','--',':','-.','-']
src=ast.parse((ROOT/'work/build_journal_plots.py').read_text());fn=next(x for x in src.body if isinstance(x,ast.FunctionDef) and x.name=='profiles')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'fixed-profiles','exec'),globals());profiles(5,5.7,'Fig_S09_all_profiles')

# Single-column alternatives use true 3.3-inch canvases with stacked panels.
SINGLE=True
for key,name,height in [('comparison','Fig_03_primary_comparison',5.0),('depth','Fig_05_stage_count',4.4),('mixing_sharing','Fig_06_mixing_sharing',4.6),('conditions','Fig_07_object_conditions',5.3)]:
    fig,axs=plt.subplots(2,1,figsize=(3.3,height))
    if key=='comparison':
        labels=['GS','HIO','HIO–ER','NCPP','Residual CNN','Encoder–decoder','SiSPRNet','FourierGSNet','Amplitude only','Shared four stages']
        for j,(ax,metric) in enumerate(zip(axs,['phase_foreground_rmse_rad','phase_foreground_ssim'])):
            for i,r in enumerate(data_log['comparison']):
                z=r[metric];m=z['mean'];lo,hi=z['ci95'];c,mk=(BLACK,'o') if i<4 else (BLUE,'s') if i<9 else (ORANGE,'D')
                ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt=mk,color=c,mfc='white' if i<4 else c,ms=3.5,capsize=2.5,lw=.8)
            ax.set_yticks(range(10),labels);ax.set_ylim(9.6,-.6);ax.axhspan(8.55,9.45,color='#f4f4f4');clean(ax);ax.set_title(['(a) Absolute phase error','(b) Structural fidelity'][j],loc='left');ax.set_xlabel('Foreground RMSE (rad)' if j==0 else 'Foreground SSIM')
        axs[0].set_xlim(0,.38);axs[0].set_xticks([0,.1,.2,.3]);axs[1].set_xlim(.4,1.01);axs[1].set_xticks([.4,.6,.8,1.])
        hh=[plt.Line2D([],[],marker=m,ls='',color=c,mfc=f,label=l) for m,c,f,l in [('o',BLACK,'white','Classical'),('s',BLUE,BLUE,'Learned baseline'),('D',ORANGE,ORANGE,'Shared four-stage')]]
        fig.legend(handles=hh,loc='upper center',ncol=1,frameon=False,bbox_to_anchor=(.62,1.0));fig.subplots_adjust(left=.43,right=.97,top=.85,bottom=.10,hspace=.48)
    elif key=='depth':
        for j,(ax,metric) in enumerate(zip(axs,['phase_foreground_rmse_rad','phase_foreground_ssim'])):
            for arms,xs,fmt,c,l in [(['stage'+str(n) for n in [1,2,3,4,6,8]],[1,2,3,4,6,8],'o-',BLUE,'Common rate (3 × 10⁻⁴)'),(['stage6_lr1e4','stage8_lr1e4'],[6,8],'s--',ORANGE,'Lower rate (10⁻⁴)')]:
                rr=[data_log['depth'][a][metric] for a in arms];m=np.array([r['mean'] for r in rr]);lo=np.array([r['ci95'][0] for r in rr]);hi=np.array([r['ci95'][1] for r in rr]);ax.errorbar(xs,m,yerr=[m-lo,hi-m],fmt=fmt,color=c,label=l,ms=3.5,capsize=2.5,lw=.8)
            clean(ax,'y');ax.axvspan(2.75,4.25,color='#f2f2f2');ax.set_xticks([1,2,3,4,6,8]);ax.set_xlabel('Reconstruction stages');ax.set_ylabel('Foreground RMSE (rad)' if j==0 else 'Foreground SSIM');ax.set_title(['(a) Absolute phase error','(b) Structural fidelity'][j],loc='left')
        hh,ll=axs[0].get_legend_handles_labels();fig.legend(hh,ll,loc='lower center',ncol=1,frameon=False);fig.subplots_adjust(left=.22,right=.97,top=.93,bottom=.19,hspace=.48)
    elif key=='mixing_sharing':
        for j,(ax,arms,ll) in enumerate(zip(axs,[mix,share],[['CNN bottleneck','FFN only','One head','Four heads','Eight heads'],['Shared','Private heads','Private decoders','Independent']])):
            for i,a in enumerate(arms):
                r=data_log['mixing_sharing'][a];m=r['mean'];lo,hi=r['ci95'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='D' if a=='stage4' else 'o',color=ORANGE if a=='stage4' else BLUE,ms=3.5,capsize=2.5,lw=.8)
            ax.set_yticks(range(len(arms)),ll);ax.set_ylim(len(arms)-.5,-.5);ax.set_xlim(.93,.96);ax.set_xticks([.93,.94,.95,.96]);ax.set_xlabel('Foreground phase SSIM');ax.set_title(['(a) Global mixing','(b) Weight sharing'][j],loc='left');clean(ax)
        hh=[plt.Line2D([],[],marker=m,ls='',color=c,label=l) for m,c,l in [('o',BLUE,'Ablation variant'),('D',ORANGE,'Shared reference')]];fig.legend(handles=hh,loc='lower center',ncol=1,frameon=False);fig.subplots_adjust(left=.39,right=.96,top=.94,bottom=.17,hspace=.48)
    else:
        groups=[['stage4','calibrated_unet','calibrated_sisprnet','fouriergs_paired','HIO','NCPP'],['stage4_coupled','stage4_mixed','unet_coupled','unet_mixed']];ll=[['Original 4-stage','Original E–D','SiSPRNet','FourierGSNet','HIO','NCPP'],['4-stage coupled','4-stage mixed','E–D coupled','E–D mixed']]
        for j,(ax,arms,labels) in enumerate(zip(axs,groups,ll)):
            for k,c in enumerate(v.CONDITIONS[1:]):
                for i,a in enumerate(arms):point(ax,A['results'][c][a]['phase_foreground_rmse_rad'],i+(k-1)*.19,COLORS[k],['o','s','^'][k])
            ax.set_yticks(range(len(arms)),labels);ax.set_ylim(len(arms)-.6,-.6);ax.set_xlim(0,1.18);ax.set_xticks([0,.5,1.]);ax.set_xlabel('Foreground phase RMSE (rad)');ax.set_title(['(a) Original training','(b) Additional training'][j],loc='left');clean(ax)
        hh=[plt.Line2D([],[],marker=m,color=c,ls='',label=l) for m,c,l in zip(['o','s','^'],COLORS,['Coupled','Different-source absorption','Pure phase'])];fig.legend(handles=hh,loc='lower center',ncol=1,frameon=False);fig.subplots_adjust(left=.36,right=.97,top=.94,bottom=.19,hspace=.42)
    save(fig,name)
SINGLE=False

for slide,name in [(1,'Fig_01_acquisition'),(2,'Fig_02_shared_reconstruction'),(3,'Fig_S02_encoder_decoder'),(2,'Fig_S03_stage_update'),(4,'Fig_S04_global_mixer'),(5,'Fig_S05_weight_sharing')]:
    with Image.open(W/f'diagram-{slide}.png') as im:
        im=im.convert('RGB');im.save(OUT/(name+'.png'),dpi=(600,600))
    receipt[name]={'width_in':5.25,'height_in':3.5,'font':'Times New Roman','minimum_text_pt':6.5,'main_text_pt':'8–9','type':'Native editable PowerPoint diagram, exported by PowerPoint','data_unchanged':True}
(W/'figure_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
(W/'plotted_statistics.json').write_text(json.dumps(data_log,indent=2),encoding='utf8')
print('Created',len(receipt),'figure layouts, including four true single-column alternatives; no fitting or inference.',flush=True)
