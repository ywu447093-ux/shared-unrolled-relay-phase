"""Fixed-budget paired adaptation of the authors' 2026 source-known FourierGSNet.

The original single-seed suite is reused without changing its training code.
This is an acquisition adaptation, not a reproduction of the published scores.
"""
import argparse,json,time
from types import SimpleNamespace
import torch,numpy as np
import suite as s
import reviewer_validation as v
from fouriergs_paired import PairedFourierGSNet,VENDOR

OUT=s.ROOT/'work/fouriergs_v1'
ARM='fouriergs_paired'

def protocol():
    OUT.mkdir(exist_ok=True)
    cfg={
      'status':'frozen before fitting or inspecting baseline scores',
      'method':'Author-supplied source-known FourierGSNet, paired relay-screen adaptation',
      'repository':'https://github.com/yanshengyuan/FourierGSNet',
      'commit':'e4f74bc720ab344d2f427f7f27968c5eff4eca0b',
      'native_variant':'Simple_forward_path-Coherent_Diffractive_Imaging-FFT/FourierGSNet_FFT_SourceKnown',
      'architecture':'11 independent original conditional U-Nets; native trunk layers retained',
      'interface_changes':'Replace native FFT by calibrated paired propagation; full 128-grid calibrated intensity replaces hard-coded 762-grid crop and resize. A bounded paired amplitude/phase fusion receives auxiliary amplitude, flat, support, saturation and geometry. Native alternating field and learned-phase states remain separate.',
      'initialization':'random, seed101; no author-trained weights',
      'training':'Original 40-epoch suite, 1184 observations/epoch, effective batch8, microbatch2, AdamW peak3e-4 and original final-output objective; same source partitions, category score and augmentation distribution. Microbatch changes joint D4 draws within each effective batch.',
      'selection':'Only original 64-content validation partition; no secondary or stress target selection',
      'budget':'Equal example and optimizer-step budget, not equal compute or demonstrated convergence',
      'evaluations':['original internal cohort',*v.CONDITIONS],
      'interpretation':'Published native scores are not used. The repository states that this source-known variant is supplied for adaptation and was not one of its reported experiments.',
      'hashes':{'suite':s.sha(s.__file__),'runner':s.sha(__file__),
        'wrapper':s.sha(s.HERE/'fouriergs_paired.py'),
        'vendor_model':s.sha(VENDOR/'model_GSNet.py'),
        'vendor_license':s.sha(VENDOR/'LICENSE'),
        'source_split':s.sha(s.OLD/'outputs/R1_controlled_experiments_v2/source_split.json')}
    }
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:s.dump(path,cfg)

def evaluate_secondary():
    ck=torch.load(OUT/'learning'/ARM/'best.pt',map_location='cpu',weights_only=False)
    model=PairedFourierGSNet().cuda().eval();model.load_state_dict(ck['model'])
    for condition in v.CONDITIONS:
        folder=OUT/'evaluation'/condition/ARM;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():continue
        cached=torch.load(v.OUT/(condition+'_cache.pt'),map_location='cpu',weights_only=False)
        rows=[];preds=[]
        with torch.inference_mode():
            for off in range(0,len(cached['records']),2):
                b=[t[off:off+2].cuda() for t in cached['data']]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
                assert torch.isfinite(pred).all()
                for j,r in enumerate(cached['records'][off:off+2]):
                    vv=s.compute_validation_metrics(pred[j:j+1],b[7][j:j+1],b[5][j:j+1],b[3][j:j+1],b[4][j:j+1])
                    mm=dict(zip(s.METRIC_NAMES,map(float,vv)));mm.update(v.phase_metrics(pred[j:j+1,1:2],b[7][j:j+1,1:2]))
                    rows.append({**r,'metrics':mm})
                preds.append(pred.cpu())
        macro={k:float(np.mean([np.mean([r['metrics'][k] for r in rows if r['cell_type']==c]) for c in sorted({r['cell_type'] for r in rows})])) for k in rows[0]['metrics']}
        s.save(folder/'predictions.pt',{'names':[r['name'] for r in rows],'predictions':torch.cat(preds)})
        s.dump(folder/'per_observation_metrics.json',rows)
        s.dump(folder/'complete.json',{'status':'complete','macro':macro,'observations':210,'source_contents':105,'checkpoint_sha256':s.sha(OUT/'learning'/ARM/'best.pt')})
        print('SECONDARY',condition,macro,flush=True)
    del model;torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-for-corrective',action='store_true');args=ap.parse_args()
    protocol()
    if args.wait_for_corrective:
        marker=s.ROOT/'work/reviewer_training_v1/all_complete.json'
        print('QUEUED: waiting for corrective experiment to finish before using GPU',flush=True)
        while not marker.exists():time.sleep(15)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original=s.model_for
    s.model_for=lambda arm:PairedFourierGSNet() if arm==ARM else original(arm)
    s.LABELS[ARM]='FourierGSNet source-known paired adaptation'
    s.OUT=OUT
    data,split,records,groups=s.load_data()
    s.train(ARM,SimpleNamespace(seed=101,epochs=40,lr=3e-4,microbatch=2),data,split,records,groups)
    del data,groups
    evaluate_secondary()
    s.dump(OUT/'all_complete.json',{'status':'complete','arm':ARM,'epochs':40,'seed':101})

if __name__=='__main__':main()
