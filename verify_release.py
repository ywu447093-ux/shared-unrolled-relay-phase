from pathlib import Path
import argparse,hashlib,json,sys
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
rows=json.loads((root/'SOURCE_INDEX.json').read_text(encoding='utf-8'))['files']
for row in rows:
 p=root/row['source'];assert p.is_file(),str(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
for p in root.rglob('*'):
 if '.git' in p.parts:continue
 assert p.suffix.lower() not in ['.pt','.pth','.ckpt','.npy','.npz','.tif','.tiff','.png','.jpg','.pdf','.docx','.zip'],str(p)
print('Verified',len(rows),'preserved scientific source files; no data or weights packaged.')
if args.smoke:
 import torch
 sys.path.insert(0,str(root/'work/experiments'))
 import suite
 torch.set_num_threads(2);torch.manual_seed(101)
 model=suite.model_for('stage4').eval()
 n=128;obs=torch.rand(1,1,n,n);pn=torch.zeros(1,3)
 pp=torch.tensor([[532e-9,1.0,.3,1e-6,100.,.02,50000.]])
 amp=torch.ones(1,1,n,n)*.8;flat=torch.ones_like(amp);support=torch.ones_like(amp);sat=torch.zeros_like(amp)
 with torch.inference_mode():pred=model([obs,pn,pp,amp,flat,support,sat])
 assert pred.shape==(1,2,n,n) and torch.isfinite(pred).all()
 assert sum(p.numel() for p in model.parameters())==1396998
 print('Synthetic forward smoke passed; parameters=1396998. No paper accuracy is inferred.')
