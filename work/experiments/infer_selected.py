"""Infer selected models on packaged observations without full training caches."""
from pathlib import Path
import argparse,json
import torch
import suite as s
import stability
stability.register()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--method',choices=s.ARMS+['GS','HIO','HIO_ER','NCPP','PhysenNet'],default='stage4')
 ap.add_argument('--examples',type=Path,default=s.ROOT/'outputs/single_seed_experiments/example_inputs.pt')
 ap.add_argument('--device',choices=['cuda','cpu'],default='cuda' if torch.cuda.is_available() else 'cpu')
 ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 obj=torch.load(args.examples,map_location='cpu',weights_only=True)
 # References are deliberately kept outside the forward input list.
 batches=[x.to(args.device) for x in obj['observations']];assert len(batches)==7
 if args.method in s.ARMS:
  model=s.model_for(args.method).to(args.device).eval()
  checkpoint=torch.load(s.OUT/'learning'/args.method/'best.pt',map_location='cpu',weights_only=False)
  model.load_state_dict(checkpoint['model']);outputs=[]
  with torch.inference_mode():
   for off in range(0,len(batches[0]),4):
    inputs=[x[off:off+4] for x in batches]
    with torch.autocast(args.device,dtype=torch.bfloat16,enabled=args.device=='cuda'):outputs.append(model(inputs).float().cpu())
  prediction=torch.cat(outputs)
 elif args.method=='PhysenNet':
  if args.device!='cuda':raise SystemExit('The per-observation fitting runner requires CUDA; supervised and classical examples also support CPU diagnostics.')
  import physennet
  config=json.loads((s.OUT/'classical/PhysenNet/complete.json').read_text())['config']
  outputs=[]
  for i in range(len(batches[0])):
   fitted,_=physennet.fit([x[i:i+1] for x in batches],[config['iterations']]);outputs.append(fitted[config['iterations']])
  prediction=torch.cat(outputs)
 else:
  import classical
  config=json.loads((s.OUT/'classical'/args.method/'complete.json').read_text())['config']
  prediction,_=classical.reconstruct(batches,args.method,config);prediction=prediction.cpu()
 assert torch.isfinite(prediction).all()
 args.output.parent.mkdir(parents=True,exist_ok=True)
 s.save(args.output,{'method':args.method,'names':obj['names'],'predictions':prediction,
  'precision':'BF16 neural / FP32 physical' if args.device=='cuda' else 'CPU FP32 diagnostic',
  'references_used_in_forward':False})
 print(json.dumps({'output':str(args.output),'method':args.method,'observations':len(prediction),'device':args.device}))

if __name__=='__main__':main()
