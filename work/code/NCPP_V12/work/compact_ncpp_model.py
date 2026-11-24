"""Compact NCPP: three physical updates and one shared context CNN.

No target phase enters forward(). Support is an optional, amplitude-derived
input feature, never a hard spatial projection. Archived models are unchanged.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import r1_controlled_experiments as legacy
from benchmark_ncpp_runtime import make_propagators
from real_model import _camera_modulus_gain, PhysicsUnrolledGenerator


class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.SiLU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+.1*self.net(x)


class ContextPrior(nn.Module):
    """One encoder-decoder, without parallel detail path or normalization."""
    def __init__(self,width=32):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(9,width,3,padding=1),nn.SiLU(),Residual(width))
        self.down1=nn.Sequential(nn.Conv2d(width,2*width,3,padding=1),nn.SiLU(),Residual(2*width))
        self.down2=nn.Sequential(nn.Conv2d(2*width,3*width,3,padding=1),nn.SiLU(),Residual(3*width),Residual(3*width))
        self.up1=nn.Sequential(nn.Conv2d(5*width,2*width,3,padding=1),nn.SiLU(),Residual(2*width))
        self.up2=nn.Sequential(nn.Conv2d(3*width,width,3,padding=1),nn.SiLU(),Residual(width))
        self.head=nn.Conv2d(width,2,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        f0=self.stem(x);f1=self.down1(F.avg_pool2d(f0,2));f2=self.down2(F.avg_pool2d(f1,2))
        u1=self.up1(torch.cat([F.interpolate(f2,size=f1.shape[-2:],mode='bilinear',align_corners=False),f1],1))
        u2=self.up2(torch.cat([F.interpolate(u1,size=f0.shape[-2:],mode='bilinear',align_corners=False),f0],1))
        return self.head(u2)


class CompactNCPP(nn.Module):
    def __init__(self,stages=3,width=32):
        super().__init__();self.stages=stages;self.prior=ContextPrior(width)
        self.dc_logits=nn.Parameter(torch.full((stages,),math.log(.15/.85)))
    def forward(self,s,pn,pp,amplitude_measurement,amplitude_flat_field,support_mask,saturation_mask,**kwargs):
        a=amplitude_measurement.clamp(0,1.25);flat=amplitude_flat_field.clamp_min(1e-4)
        support=support_mask.clamp(0,1);phase=torch.zeros_like(a);amp=a
        prop,inv=make_propagators(pp,s.shape[-1]);gain=_camera_modulus_gain(pp)
        lo=pp[:,3:4,None,None];span=(pp[:,4:5]-pp[:,3:4])[:,:,None,None]
        measured=(s*span+lo)/gain
        absorption=((flat-a)/flat).clamp(-.1,.15)*20
        maps=pn[:,:3,None,None].expand(-1,-1,*a.shape[-2:])
        for k in range(self.stages):
            progress=k/max(self.stages-1,1);gamma=1.2-.4*progress
            field=prop(torch.polar(amp,phase));modulus=field.abs()
            candidate=(s.clamp(0,1).pow(gamma)*span+lo)/gain
            candidate=torch.where((saturation_mask>.5)&(modulus>measured),modulus,candidate)
            obj=inv(candidate*field/(modulus+1e-8))
            phase_dc=torch.angle(obj).clamp(0,math.pi)
            features=torch.cat([phase/math.pi,phase_dc/math.pi,
                ((obj.abs()-a)/flat).clamp(-2,2),absorption,support,amplitude_flat_field,maps],1)
            delta=self.prior(features)
            phase=(phase+torch.sigmoid(self.dc_logits[k])*(phase_dc-phase)+.5*math.pi*torch.tanh(delta[:,1:2])).clamp(0,math.pi)
            amp=(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25)
        return torch.cat([amp,phase],1)


class ZeroRefiner(nn.Module):
    def forward(self,x):return torch.zeros_like(x[:,:1])


def model_for(arm):
    if arm=='compact3':return CompactNCPP()
    if arm=='lean3':
        m=PhysicsUnrolledGenerator(num_stages=3,features=32,img_size=128,shared_prior=True)
        m.prior=legacy.BranchPrior(m.prior,'no_detail')
        m.output_refiner=ZeroRefiner()
        return m
    raise ValueError(arm)
