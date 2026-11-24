"""Inference from packaged observations with selected evidence-study checkpoints."""
import argparse,json
from pathlib import Path
import torch
import suite as s
import calibrated
from fouriergs_paired import PairedFourierGSNet
from no_diffraction_train import NoDiffraction

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',default='stage4_mixed',choices=['stage4','stage3','no_global','unshared','calibrated_unet','calibrated_sisprnet','amplitude_only','stage4_mixed','stage4_coupled','unet_mixed','unet_coupled','fouriergs_paired','stage4_no_diffraction'])
    ap.add_argument('--condition',default='independent_absorption',choices=['secondary','fresh_coupled','independent_absorption','pure_phase'])
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size',type=int,default=None,help='Default replays the archived evaluation batches; one permits singleton diagnostics.')
    ap.add_argument('--output',type=Path,default=s.ROOT/'work/evidence_example_predictions.pt')
    args=ap.parse_args();torch.set_num_threads(4);calibrated.register()
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    method=args.method
    if method in ['stage4_mixed','stage4_coupled','unet_mixed','unet_coupled']:
        model=s.model_for('stage4' if method.startswith('stage4') else 'calibrated_unet');checkpoint=s.ROOT/'work/reviewer_training_v1'/method/'best.pt'
    elif method=='fouriergs_paired':model=PairedFourierGSNet();checkpoint=s.ROOT/'work/fouriergs_v1/learning'/method/'best.pt'
    elif method=='stage4_no_diffraction':model=NoDiffraction();checkpoint=s.ROOT/'work/no_diffraction_v1/learning'/method/'best.pt'
    else:model=s.model_for(method);checkpoint=s.ROOT/'work/experiment_results_v1/learning'/method/'best.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.to(args.device).eval()
    cachefolder='balanced_pairing_v1' if args.condition=='independent_absorption' else 'reviewer_validation_v1'
    cache=torch.load(s.ROOT/'work'/cachefolder/(args.condition+'_cache.pt'),map_location='cpu',weights_only=False)
    categories=sorted({r['cell_type'] for r in cache['records']});ids=[next(i for i,r in enumerate(cache['records']) if r['cell_type']==c) for c in categories]
    # Reference fields stay outside the forward argument.
    batch_size=args.batch_size or (2 if method=='fouriergs_paired' else 4)
    assert batch_size>0
    predictions={}
    with torch.inference_mode():
        for off in sorted({i//batch_size*batch_size for i in ids}):
            observations=[x[off:off+batch_size].to(args.device) for x in cache['data'][:7]]
            with torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,enabled=torch.device(args.device).type=='cuda'):
                prediction=model(observations).float().cpu()
            for i in ids:
                if off<=i<off+len(prediction):predictions[i]=prediction[i-off:i-off+1]
    pred=torch.cat([predictions[i] for i in ids]);assert pred.shape==(5,2,128,128) and torch.isfinite(pred).all()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'method':method,'condition':args.condition,'batch_size':batch_size,'records':[cache['records'][i] for i in ids],'predictions':pred,'references':cache['data'][7][ids],'reference_note':'stored for evaluation only; excluded from forward inputs'},args.output)
    print(json.dumps({'output':str(args.output),'observations':5,'shape':list(pred.shape),'finite':True,'checkpoint_sha256':s.sha(checkpoint)}),flush=True)

if __name__=='__main__':main()
