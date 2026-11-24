"""Stage-conditioned shared CNN; no extra branch, encoder or normalization."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from compact_ncpp_model import ContextPrior
from compact_ncpp_v8_model import ConfidenceNCPP
from benchmark_ncpp_runtime import make_propagators
from real_model import _camera_modulus_gain

ARMS = ['continue48', 'stage3', 'stage4']


class StageContextPrior(ContextPrior):
    def __init__(self, width=48):
        super().__init__(width)
        self.stage_bias = nn.Parameter(torch.zeros(width))

    def forward(self, features, progress):
        # A known iteration coordinate shifts the existing stem activation.
        # Zero initialization preserves the pretrained three-step function.
        stem = self.stem[0](features) + progress * self.stage_bias[None, :, None, None]
        f0 = self.stem[2](self.stem[1](stem))
        f1 = self.down1(F.avg_pool2d(f0, 2))
        f2 = self.down2(F.avg_pool2d(f1, 2))
        u1 = self.up1(torch.cat([F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False), f1], 1))
        u2 = self.up2(torch.cat([F.interpolate(u1, size=f0.shape[-2:], mode='bilinear', align_corners=False), f0], 1))
        return self.head(u2)


class StageConditionedNCPP(ConfidenceNCPP):
    def __init__(self, stages=3, width=48):
        super().__init__(stages=stages, width=width)
        self.prior = StageContextPrior(width)

    def forward(self, s, pn, pp, amplitude_measurement, amplitude_flat_field, support_mask, saturation_mask, **kwargs):
        a = amplitude_measurement.clamp(0, 1.25)
        flat = amplitude_flat_field.clamp_min(1e-4)
        phase, amp = torch.zeros_like(a), a
        prop, inv = make_propagators(pp, s.shape[-1])
        gain = _camera_modulus_gain(pp)
        lo = pp[:, 3:4, None, None]
        span = (pp[:, 4:5] - pp[:, 3:4])[:, :, None, None]
        measured = (s * span + lo) / gain
        quantiles = torch.quantile(s.detach().flatten(2), torch.tensor([.08, .30], device=s.device, dtype=s.dtype), dim=2)
        floor = quantiles[0, :, :, None, None]
        sigma = (quantiles[1] - quantiles[0]).clamp_min(.005)[:, :, None, None]
        signal = (s - floor).clamp_min(0)
        weight = .25 + .75 * signal / (signal + sigma + 1e-8)
        absorption = ((flat - a) / flat).clamp(-.1, .15) * 20
        maps = pn[:, :3, None, None].expand(-1, -1, *a.shape[-2:])
        for k in range(self.stages):
            u = torch.polar(amp, phase)
            field = prop(u)
            modulus = field.abs()
            bound = torch.where((saturation_mask > .5) & (modulus > measured), modulus, measured)
            candidate = modulus + weight * (bound - modulus)
            obj = inv(candidate * field / (modulus + 1e-8))
            residual = (obj - u) / flat
            step = torch.sin(torch.angle(obj * torch.conj(u)))
            features = torch.cat([phase / math.pi, residual.real.clamp(-2, 2), residual.imag.clamp(-2, 2),
                                  absorption, support_mask.clamp(0, 1), amplitude_flat_field, maps], 1)
            delta = self.prior(features, k / max(self.stages - 1, 1))
            phase = (phase + torch.sigmoid(self.dc_logits[k]) * step + .5 * math.pi * torch.tanh(delta[:, 1:2])).clamp(0, math.pi)
            amp = (a + .03 * torch.tanh(delta[:, :1])).clamp(0, 1.25)
        return torch.cat([amp, phase], 1)


def model_for(arm):
    assert arm in ARMS
    return ConfidenceNCPP(width=48) if arm == 'continue48' else StageConditionedNCPP(stages=3 if arm == 'stage3' else 4)


def initialize(arm, path):
    model = model_for(arm)
    state = dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    if arm == 'stage4':
        state['dc_logits'] = torch.cat([state['dc_logits'], state['dc_logits'][-1:]])
    missing = model.load_state_dict(state, strict=False)
    assert missing.unexpected_keys == []
    assert missing.missing_keys == ([] if arm == 'continue48' else ['prior.stage_bias'])
    return model
