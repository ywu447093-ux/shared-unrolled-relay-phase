"""Measurement-only confidence relaxation; same three-step shared CNN."""
import math
import torch
from compact_ncpp_v3_model import ComplexResidualNCPP
from benchmark_ncpp_runtime import make_propagators
from real_model import _camera_modulus_gain

ARMS=['continue48','confidence48','confidence_low48']

class ConfidenceNCPP(ComplexResidualNCPP):
    def forward(self,s,pn,pp,amplitude_measurement,amplitude_flat_field,support_mask,saturation_mask,**kwargs):
        a=amplitude_measurement.clamp(0,1.25);flat=amplitude_flat_field.clamp_min(1e-4)
        phase=torch.zeros_like(a);amp=a;prop,inv=make_propagators(pp,s.shape[-1]);gain=_camera_modulus_gain(pp)
        lo=pp[:,3:4,None,None];span=(pp[:,4:5]-pp[:,3:4])[:,:,None,None];measured=(s*span+lo)/gain
        quantiles=torch.quantile(s.detach().flatten(2),torch.tensor([.08,.30],device=s.device,dtype=s.dtype),dim=2)
        floor=quantiles[0,:,:,None,None];sigma=(quantiles[1]-quantiles[0]).clamp_min(.005)[:,:,None,None]
        signal=(s-floor).clamp_min(0);weight=.25+.75*signal/(signal+sigma+1e-8)
        absorption=((flat-a)/flat).clamp(-.1,.15)*20;maps=pn[:,:3,None,None].expand(-1,-1,*a.shape[-2:])
        for k in range(self.stages):
            u=torch.polar(amp,phase);field=prop(u);modulus=field.abs()
            bound=torch.where((saturation_mask>.5)&(modulus>measured),modulus,measured)
            candidate=modulus+weight*(bound-modulus)
            obj=inv(candidate*field/(modulus+1e-8));residual=(obj-u)/flat
            step=torch.sin(torch.angle(obj*torch.conj(u)))
            features=torch.cat([phase/math.pi,residual.real.clamp(-2,2),residual.imag.clamp(-2,2),absorption,support_mask.clamp(0,1),amplitude_flat_field,maps],1)
            delta=self.prior(features)
            phase=(phase+torch.sigmoid(self.dc_logits[k])*step+.5*math.pi*torch.tanh(delta[:,1:2])).clamp(0,math.pi)
            amp=(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25)
        return torch.cat([amp,phase],1)

def model_for(arm):
    assert arm in ARMS
    return ComplexResidualNCPP(width=48) if arm=='continue48' else ConfidenceNCPP(width=48)

def initialize(arm,path):
    m=model_for(arm);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return m
