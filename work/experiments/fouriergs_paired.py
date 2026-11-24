"""Author-supplied source-known FourierGSNet architecture adapted to paired data.

The MIT-licensed conditional U-Net is loaded without changing its layers. Only
the two deterministic training-time einops reshapes are supplied by torch.
The physical interface uses the existing calibrated paired Fresnel propagator.
"""
import ast,types,time,json,math,argparse
from pathlib import Path
import torch
from torch import nn
import suite as s
VENDOR=s.ROOT/'work/reviewer_validation_v1/fouriergs_vendor'

def rearrange(x,pattern,**kwargs):
    if pattern=='b (c1 c2) h w -> b c1 (c2 h w)':return x.reshape(x.shape[0],kwargs['c1'],-1)
    if pattern=='b c (h w) -> b c h w':return x.reshape(x.shape[0],x.shape[1],kwargs['h'],kwargs['w'])
    raise ValueError('Visualization-only reshape is not enabled: '+pattern)

def load_vendor():
    tree=ast.parse((VENDOR/'model_GSNet.py').read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='einops')]
    module=types.ModuleType('fouriergs_author_model');module.rearrange=rearrange
    exec(compile(tree,str(VENDOR/'model_GSNet.py'),'exec'),module.__dict__)
    return module

class PairedFourierGSNet(nn.Module):
    def __init__(self):
        super().__init__();vendor=load_vendor()
        # Published code: one input layer and ten subsequent independent layers.
        self.layers=nn.ModuleList([vendor.Condition_UNet() for _ in range(11)])
        self.fusion=nn.Sequential(nn.Conv2d(9,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,2,3,padding=1))
        rng=torch.Generator().manual_seed(101)
        self.register_buffer('gs_phase_init',(torch.rand((1,1,128,128),generator=rng)*2-1)*math.pi)
        self.register_buffer('gen_phase_init',(torch.rand((1,1,128,128),generator=rng)*2-1)*math.pi)
    def forward(self,b):
        obs,pn,pp,a,flat,support,sat=b[:7];a=a.clamp(0,1.25)
        mu=(obs*(pp[:,4:5]-pp[:,3:4])[:,:,None,None]+pp[:,3:4,None,None])/s._camera_modulus_gain(pp)
        prop,inv=s.make_propagators(pp,128)
        fftscale=(pp[:,0:1]*pp[:,1:2]/pp[:,2:3].square())[:,:,None,None]*128**2
        measured=mu/fftscale.clamp_min(1e-10)
        intensity=measured.square();field=torch.polar(a,self.gs_phase_init.expand_as(a));phase=self.gen_phase_init.expand_as(a)
        for network in self.layers:
            screen=prop(field);mod=screen.abs();bound=torch.where(sat>.5,torch.maximum(mod,mu),mu)
            back=inv(bound*screen/(mod+1e-8));physical_phase=torch.angle(back)
            field=torch.polar(a,physical_phase)
            scaled=screen/fftscale.clamp_min(1e-10)
            x=torch.cat([phase,intensity],1)
            condition=torch.cat([physical_phase,scaled.real,scaled.imag],1)
            attention=torch.cat([phase,physical_phase,scaled.real,scaled.imag],1)
            phase=network(x,condition,attention)
        absorption=((flat.clamp_min(1e-4)-a)/flat.clamp_min(1e-4)).clamp(-.1,.15)*20
        maps=pn[:,:3,None,None].expand(-1,-1,128,128)
        delta=self.fusion(torch.cat([phase,absorption,b[3],flat,support,sat,maps],1)).float()
        return torch.cat([(a+.03*torch.tanh(delta[:,:1])).clamp(0,1.25),math.pi*torch.sigmoid(delta[:,1:2])],1)

def benchmark():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;s.seed(101)
    data,split,records,_=s.load_data(False);b=[t[split['train'][:1]].cuda() for t in data]
    model=PairedFourierGSNet().cuda();physics=s.DeterministicPhysicalLoss().cuda()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(b).float()
    loss=s.objective(pred,b,physics);loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(pred).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    x=torch.arange(768*8*8).reshape(1,768,8,8)
    assert torch.equal(rearrange(rearrange(x,'b (c1 c2) h w -> b c1 (c2 h w)',c1=3),'b c (h w) -> b c h w',h=128,w=128),x.reshape(1,3,128,128))
    record={'parameters':sum(p.numel() for p in model.parameters()),'one_image_forward_backward_s':time.time()-start,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'loss':float(loss),'finite_gradients':True,'reshape_equivalence_checked':True,'vendor_model_sha256':s.sha(VENDOR/'model_GSNet.py'),'wrapper_sha256':s.sha(__file__)}
    s.dump(s.ROOT/'work/reviewer_validation_v1/fouriergs_smoke.json',record);print(record,flush=True)

if __name__=='__main__':benchmark()
