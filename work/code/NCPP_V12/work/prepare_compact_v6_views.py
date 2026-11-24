"""Generate native train-parent views with unchanged audited optical simulation."""
import hashlib,json,msvcrt,os,time,traceback
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
import r1_controlled_experiments as s
import prepare_real_dataset as gen
from real_GAN_dataset import RelaySPSDataset
from amplitude_support import estimate_support_from_amplitude

OUT=s.ROOT/'outputs/compact_ncpp_v6';SEED=20261021

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def seeded(key):return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{SEED}|{key}'.encode()).digest()[:8],'little'))
def identity(a):return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def choose(image,region,rng):
    top,left,bottom,right=region
    if bottom-top<128 or right-left<128:return None
    positions=[((top+bottom-128)//2,(left+right-128)//2)]
    positions += [(int(rng.integers(top,bottom-128+1)),int(rng.integers(left,right-128+1))) for _ in range(15)]
    best=None
    for y,x in positions:
        crop=image[y:y+128,x:x+128];gx=cv2.Sobel(crop,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(crop,cv2.CV_32F,0,1,ksize=3)
        score=float(np.mean(np.hypot(gx,gy))+.2*np.std(crop))
        if best is None or score>best['score']:best={'top':y,'left':x,'score':score}
    return best

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    lock=(OUT/'generation.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('An active view generator holds the lock')
    state={'status':'generating','pid':os.getpid(),'started_unix':time.time()}
    save=lambda:s.dump(OUT/'generation_status.json',state)
    try:
        split=json.loads((s.OUT/'source_split.json').read_text(encoding='utf8'))
        parents={}
        for row in split['train']:parents.setdefault(row['content_id'],{'source_qpm':row['source_qpm'],'cell_type':row['cell_type'],'content_id':row['content_id']})
        forbidden={r['content_id'] for part in ['validation','holdout'] for r in split[part]}
        assert len(parents)==294 and not set(parents)&forbidden
        cfg={'seed':SEED,'split_sha256':s.sha(s.OUT/'source_split.json'),'protocol_sha256':s.sha(OUT/'EXPERIMENT_PROTOCOL.md'),
             'source_hashes':{str(Path(p).relative_to(s.ROOT)):s.sha(p) for p in [Path(__file__),Path(gen.__file__),s.CODE/'amplitude_support.py',s.CODE/'real_GAN_dataset.py']},
             'train_parent_ids':sorted(parents),'simulations_per_view':2,'image_size':128,'same_camera_noise_and_simulator_as_original':True}
        cp=OUT/'generation_config.json'
        if cp.exists():assert json.loads(cp.read_text())==cfg,'Generation configuration changed'
        else:s.dump(cp,cfg)
        if (OUT/'augmented_cache.pt').exists():
            previous=json.loads((OUT/'generation_complete.json').read_text());assert previous['cache_sha256']==sha_file(OUT/'augmented_cache.pt')
            print('Existing verified augmentation cache retained',flush=True);return
        save();views=[];dimensions=Counter()
        for parent in sorted(parents):
            meta=parents[parent];raw=cv2.imread(meta['source_qpm'],cv2.IMREAD_UNCHANGED)
            assert raw is not None and identity(raw)==parent and Path(meta['source_qpm']).name.endswith('_img.tif')
            image=gen._read_qpm(meta['source_qpm']);h,w=image.shape;dimensions[str((h,w))]+=1;assert min(h,w)>=128
            regions=[(0,0,h//2,w//2),(0,w//2,h//2,w),(h//2,0,h,w//2),(h//2,w//2,h,w)]
            options=[(j,choose(image,region,seeded(f'{parent}:q{j}'))) for j,region in enumerate(regions)]
            options=[(j,x) for j,x in options if x is not None];maximum=max([x['score'] for _,x in options],default=0.)
            selected=[]
            for quadrant,item in options:
                if item['score']<.25*maximum:continue
                y,x=item['top'],item['left'];crop=image[y:y+128,x:x+128]
                try:fraction=gen.qpm_to_phase_fraction(crop)
                except ValueError:continue
                if float(np.mean(fraction>.1))<.02:continue
                selected.append({**meta,**item,'quadrant':quadrant,'raw_crop_sha256':identity(crop)})
            if not selected:
                item=choose(image,(0,0,h,w),seeded(f'{parent}:fallback'));assert item is not None
                crop=image[item['top']:item['top']+128,item['left']:item['left']+128];fraction=gen.qpm_to_phase_fraction(crop)
                assert np.mean(fraction>.1)>=.02,('No usable augmentation view',parent)
                selected=[{**meta,**item,'quadrant':'fallback_full_native','raw_crop_sha256':identity(crop)}]
            views.extend(selected)
        n=2*len(views);data=[torch.empty((n,1,128,128)) if j in [0,3,4,5,6] else torch.empty((n,3 if j==1 else 7)) if j in [1,2] else torch.empty((n,2,128,128)) for j in range(8)]
        manifest=[];last_parent=None;image=None
        for vi,view in enumerate(views):
            if view['content_id']!=last_parent:image=gen._read_qpm(view['source_qpm']);last_parent=view['content_id']
            y,x=view['top'],view['left'];crop=image[y:y+128,x:x+128];assert identity(crop)==view['raw_crop_sha256']
            fraction=gen.qpm_to_phase_fraction(crop)
            for simulation in range(2):
                key=f"{view['content_id']}:{y}:{x}:{simulation}";rng=seeded(key)
                amplitude,phase,flat=gen.generate_qpm_proxy(fraction,rng);gain=float(rng.uniform(.92,1.02))
                aobs=gen.simulate_amplitude_measurement(amplitude,rng,gain=gain);fobs=gen.simulate_amplitude_measurement(flat,rng,gain=gain)
                support=estimate_support_from_amplitude(aobs,flat_field=fobs);sps,sat,params,diagnostics=gen.generate_real_sps_corrected(phase,amplitude,rng)
                norm=np.array([RelaySPSDataset._normalize(float(params[0]),525e-9,540e-9),RelaySPSDataset._normalize(float(params[1]),.85,1.15),
                               RelaySPSDataset._normalize(np.log10(max(float(params[2]),1e-8)),-.66,-.45)],dtype=np.float32)
                arrays=[sps[None],norm,params.astype(np.float32),aobs[None],fobs[None],support.soft[None],(sat>.5).astype(np.float32)[None],np.stack([amplitude,phase])]
                idx=2*vi+simulation
                for j,value in enumerate(arrays):
                    assert np.isfinite(value).all();data[j][idx].copy_(torch.from_numpy(np.ascontiguousarray(value)))
                manifest.append({**view,'name':f'aug_{idx:06d}','index':idx,'simulation_index':simulation,'rng_key':key,'physical_params_float64':list(map(float,params)),
                                 'support_source':support.source,'support_valid':support.valid,'support_uses_phase':False,'support_uses_annotation_mask':False,**diagnostics})
            if (vi+1)%25==0 or vi+1==len(views):
                state.update(views_finished=vi+1,views_total=len(views),samples_finished=2*(vi+1));save();print(json.dumps(state),flush=True)
        assert all(torch.isfinite(t).all() for t in data) and float(data[0].min())>=0 and float(data[0].max())<=1
        assert float(data[7][:,1].min())>=0 and float(data[7][:,1].max())<=np.pi+1e-6
        s.dump(OUT/'augmentation_manifest.json',manifest)
        pending=OUT/'augmented_cache.pending.pt';torch.save(data,pending);os.replace(pending,OUT/'augmented_cache.pt')
        report={'status':'verified','samples':n,'native_views':len(views),'independent_training_parent_contents':len(parents),'raw_parent_dimensions':dict(dimensions),
                'samples_by_category':dict(Counter(r['cell_type'] for r in manifest)),'views_per_parent':dict(Counter(r['content_id'] for r in views)),
                'cache_sha256':sha_file(OUT/'augmented_cache.pt'),'manifest_sha256':s.sha(OUT/'augmentation_manifest.json'),'config_sha256':s.sha(cp),
                'partition_isolation':True,'validation_and_holdout_unchanged':True,'additional_independent_parent_images':0}
        s.dump(OUT/'generation_complete.json',report);state.update(status='verified_complete',finished_unix=time.time(),samples=n,native_views=len(views));save();print(json.dumps(report),flush=True)
    except Exception:
        state.update(status='failed',error=traceback.format_exc(),finished_unix=time.time());save();raise
    finally:lock.close()

if __name__=='__main__':main()
