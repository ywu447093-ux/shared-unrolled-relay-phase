"""Full-calibration image baselines added before any direct-model results.

Retain the normalized-SPS controls as diagnostics. Primary learned baselines
receive the complete calibrated modulus normalized by the known FFT sampling
factor, so extrema/gain information is not silently omitted.
"""
import argparse,sys
import torch
from torch import nn
import suite as s

MAPPING={'calibrated_unet':'direct_unet','calibrated_rescnn':'direct_rescnn','calibrated_sisprnet':'sisprnet_adapted'}
_factory=s.model_for

class CalibratedInput(nn.Module):
 def __init__(self,arm):
  super().__init__();self.base=_factory(MAPPING[arm])
  if arm=='calibrated_sisprnet':self.base.fusion[0]=nn.Conv2d(10,48,3,padding=1)
  elif arm=='calibrated_rescnn':self.base.net[0]=nn.Conv2d(10,64,3,padding=1)
  else:self.base.net.stem[0]=nn.Conv2d(10,48,3,padding=1)
 def forward(self,b):
  pp=b[2];mu=(b[0]*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/s._camera_modulus_gain(pp)
  fft_scale=(pp[:,0:1]*pp[:,1:2]/pp[:,2:3].square())[:,:,None,None]*b[0].shape[-1]**2
  measured=mu/fft_scale.clamp_min(1e-10)
  a=b[3].clamp(0,1.25);flat=b[4];support=b[5];sat=b[6]
  absorption=((flat.clamp_min(1e-4)-a)/flat.clamp_min(1e-4)).clamp(-.1,.15)*20
  maps=b[1][:,:3,None,None].expand(-1,-1,*a.shape[-2:])
  if self.base.arm=='sisprnet_adapted':
   z=self.base.backbone(measured)
   delta=self.base.fusion(torch.cat([z,absorption,b[3],flat,support,sat,maps],1)).float()
  else:
   compressed=torch.log1p(100*measured)/s.math.log(101)
   x=torch.cat([measured,compressed,absorption,b[3],flat,support,sat,maps],1)
   delta=(self.base.net(x) if self.base.arm=='direct_rescnn' else self.base.net(x,0.)).float()
  return torch.cat([(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25),s.math.pi*torch.sigmoid(delta[:,1:2])],1)

def register():
 if getattr(s,'calibrated_models_registered',False):return
 def factory(arm):return CalibratedInput(arm) if arm in MAPPING else _factory(arm)
 s.model_for=factory;s.ARMS+=list(MAPPING)
 s.LABELS.update({'calibrated_unet':'Calibrated direct encoder-decoder','calibrated_rescnn':'Calibrated direct residual CNN','calibrated_sisprnet':'Calibrated SiSPRNet adaptation'})
 s.calibrated_models_registered=True

def main():
 register();s.dump(s.OUT/'calibrated_input_amendment.json',{
  'reason':'Ensure primary direct baselines receive full measured modulus calibration rather than only min-max-normalized SPS',
  'decision_timing':'before training or inspecting any direct learned baseline result',
  'transformation':'mu / ((lambda*z/Ls^2)*N^2), invertible given supplied optical parameters',
  'weak_absorption_conditioning':'explicit 20*clip((floored_flat-clipped_amplitude)/floored_flat,-0.1,0.15), also supplied to the proposed model; no original information removed',
  'feature_design':'CNN/encoder-decoder retains both calibrated modulus and compressed modulus and adds absorption (10 channels); SiSPRNet adds absorption to its auxiliary fusion (10 channels)',
  'wrapper_sha256':s.sha(__file__),'vendor_sha256':s.sha(s.HERE/'vendor/model_SiPRNet_original.py'),
  'budget':'same seed 101, 40 epochs, optimizer, objective, order and paired observations as other supervised runs',
  'retention':'normalized-SPS controls retained; calibrated versions are the primary equal-information comparison'})
 sys.argv=[str(s.HERE/'suite.py'),'--epochs','40','--seed','101','--arms',*MAPPING]
 s.main()
if __name__=='__main__':main()
