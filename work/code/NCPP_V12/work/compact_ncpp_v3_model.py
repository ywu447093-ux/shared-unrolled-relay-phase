"""Three compact .95 candidates; archived model sources remain immutable."""
import math
import torch
from torch import nn
from compact_ncpp_model import CompactNCPP
from benchmark_ncpp_runtime import make_propagators
from real_model import _camera_modulus_gain

ARMS=['aligned32','complex32','wide48']

class ComplexResidualNCPP(CompactNCPP):
    """Same shared CNN; untruncated complex residuals and true final modulus."""
    def forward(self,s,pn,pp,amplitude_measurement,amplitude_flat_field,support_mask,saturation_mask,**kwargs):
        a=amplitude_measurement.clamp(0,1.25);flat=amplitude_flat_field.clamp_min(1e-4)
        phase=torch.zeros_like(a);amp=a;prop,inv=make_propagators(pp,s.shape[-1]);gain=_camera_modulus_gain(pp)
        lo=pp[:,3:4,None,None];span=(pp[:,4:5]-pp[:,3:4])[:,:,None,None]
        measured=(s*span+lo)/gain
        absorption=((flat-a)/flat).clamp(-.1,.15)*20
        maps=pn[:,:3,None,None].expand(-1,-1,*a.shape[-2:])
        for k in range(self.stages):
            u=torch.polar(amp,phase);field=prop(u);modulus=field.abs()
            # Exact calibrated modulus: no heuristic gamma distortion.
            candidate=torch.where((saturation_mask>.5)&(modulus>measured),modulus,measured)
            obj=inv(candidate*field/(modulus+1e-8))
            residual=(obj-u)/flat
            # The sine phase gradient is continuous at the +/-pi branch cut.
            # It is the negative phase direction of complex-field mismatch.
            phase_step=torch.sin(torch.angle(obj*torch.conj(u)))
            features=torch.cat([phase/math.pi,residual.real.clamp(-2,2),residual.imag.clamp(-2,2),
                                absorption,support_mask.clamp(0,1),amplitude_flat_field,maps],1)
            delta=self.prior(features)
            phase=(phase+torch.sigmoid(self.dc_logits[k])*phase_step+.5*math.pi*torch.tanh(delta[:,1:2])).clamp(0,math.pi)
            amp=(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25)
        return torch.cat([amp,phase],1)

def model_for(arm):
    if arm=='aligned32':return CompactNCPP()
    if arm=='complex32':return ComplexResidualNCPP()
    if arm=='wide48':return ComplexResidualNCPP(width=48)
    raise ValueError(arm)

def widen_state(old,new):
    """Net2Wider-style duplication with fan-in compensation, no extra blocks."""
    old_convs={k:v for k,v in old.named_modules() if isinstance(v,nn.Conv2d)}
    new_convs={k:v for k,v in new.named_modules() if isinstance(v,nn.Conv2d)}
    maps={key:torch.arange(newc)%oldc for key,oldc,newc in [('s',32,48),('f1',64,96),('f2',96,144)]}
    for name,dst in new_convs.items():
        src=old_convs[name]
        if name=='prior.stem.0':im=torch.arange(9);om=maps['s']
        elif name.startswith('prior.stem.'):im=om=maps['s']
        elif name=='prior.down1.0':im=maps['s'];om=maps['f1']
        elif name.startswith('prior.down1.'):im=om=maps['f1']
        elif name=='prior.down2.0':im=maps['f1'];om=maps['f2']
        elif name.startswith('prior.down2.'):im=om=maps['f2']
        elif name=='prior.up1.0':im=torch.cat([maps['f2'],maps['f1']+96]);om=maps['f1']
        elif name.startswith('prior.up1.'):im=om=maps['f1']
        elif name=='prior.up2.0':im=torch.cat([maps['f1'],maps['s']+64]);om=maps['s']
        elif name.startswith('prior.up2.'):im=om=maps['s']
        elif name=='prior.head':im=maps['s'];om=torch.arange(2)
        else:raise ValueError(name)
        counts=torch.bincount(im,minlength=src.in_channels)
        with torch.no_grad():
            dst.weight.copy_(src.weight[om][:,im]/counts[im][None,:,None,None])
            dst.bias.copy_(src.bias[om])
            # Break duplicate-channel learning symmetry without adding a
            # forward signal: paired outgoing perturbations sum to zero.
            for j in torch.where(counts==2)[0]:
                pair=torch.where(im==j)[0]
                noise=torch.randn_like(dst.weight[:,pair[0]])*(.01*src.weight.std())
                dst.weight[:,pair[0]].add_(noise)
                dst.weight[:,pair[1]].sub_(noise)
    with torch.no_grad():new.dc_logits.copy_(old.dc_logits)

def initialize(arm,checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);old=CompactNCPP();old.load_state_dict(ck['model'])
    m=model_for(arm)
    if arm=='wide48':widen_state(old,m)
    else:m.load_state_dict(old.state_dict())
    if arm in ['complex32','wide48']:
        # The two residual channels have changed meaning. Do not silently
        # reuse phase/absolute-modulus feature filters as complex filters.
        with torch.no_grad():m.prior.stem[0].weight[:,1:3].zero_()
    return m
