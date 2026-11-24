"""Replace existing bottleneck residual blocks with one nonlocal mixer."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from compact_ncpp_model import Residual
from compact_ncpp_v11_model import model_for as previous_model

ARMS = ['continue48', 'global_one48', 'global_two48']
EXPECTED_PARAMETERS = {'continue48': 1603062, 'global_one48': 1396998, 'global_two48': 1023462}
EXPECTED_CONV_GMAC = {'continue48': 22.904045568, 'global_one48': 22.054699008, 'global_two48': 20.5258752}
EXPECTED_ATTENTION_GMAC = {'continue48': 0., 'global_one48': 1.207959552, 'global_two48': 1.207959552}


class GlobalMixingBlock(nn.Module):
    """One pre-normalized, four-head spatial mixer plus pointwise MLP.

    Operates at the existing 32x32,144-channel bottleneck. No parallel branch,
    positional embeddings, dropout, extra encoder or extra physical update.
    """
    def __init__(self, channels=144, heads=4):
        super().__init__()
        assert channels % heads == 0
        self.channels, self.heads = channels, heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Conv2d(channels, 3*channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 2*channels, 1), nn.SiLU(), nn.Conv2d(2*channels, channels, 1))
        # Small NONZERO outputs keep gradients connected at the first step.
        # Removed pretrained residual blocks are not reproduced at transfer.
        for layer in [self.projection, self.ffn[2]]:
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def normalized(norm, x):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def attention_message(self, x):
        batch, channels, height, width = x.shape
        tokens = height*width
        values = self.qkv(self.normalized(self.norm1, x)).reshape(batch, 3, self.heads, channels//self.heads, tokens)
        query, key, value = [v.transpose(-2, -1).contiguous() for v in values.unbind(1)]
        # Fix the backend as part of the reproducibility protocol; never use
        # an automatic fused-kernel choice or train/eval-dependent dropout.
        with sdpa_kernel(SDPBackend.MATH):
            mixed = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        return mixed.transpose(-2, -1).reshape(batch, channels, height, width)

    def forward(self, x):
        x = x + .1*self.projection(self.attention_message(x))
        return x + .1*self.ffn(self.normalized(self.norm2, x))


def model_for(arm):
    assert arm in ARMS
    model = previous_model('dilate_wide48')
    if arm == 'global_one48':
        model.prior.down2[3] = GlobalMixingBlock()
    elif arm == 'global_two48':
        model.prior.down2[2] = GlobalMixingBlock()
        model.prior.down2[3] = nn.Identity()
    return model


def architecture(model):
    blocks = [{'name': name, 'channels': block.channels, 'heads': block.heads,
               'normalization': 'LayerNorm across channels per spatial token',
               'backend': 'SDPBackend.MATH', 'dropout': 0., 'ffn_expansion': 2,
               'residual_scale': .1}
              for name, block in model.named_modules() if isinstance(block, GlobalMixingBlock)]
    retained = {str(index): [model.prior.down2[index].net[j].dilation[0] for j in [0, 2]]
                for index in [2, 3] if isinstance(model.prior.down2[index], Residual)}
    return {'stages': model.stages, 'width': 48, 'global_modules': blocks,
            'retained_bottleneck_dilations': retained, 'stage_conditioning': True}


def initialize(arm, path):
    model = model_for(arm)
    old = torch.load(path, map_location='cpu', weights_only=False)['model']
    current = model.state_dict()
    expected_prefixes = {'continue48': (), 'global_one48': ('prior.down2.3.',),
                         'global_two48': ('prior.down2.2.', 'prior.down2.3.')}[arm]
    removed = set(old)-set(current)
    expected_removed = {key for key in old if key.startswith(expected_prefixes)} if expected_prefixes else set()
    assert removed == expected_removed
    new = set(current)-set(old)
    assert all(key.startswith(expected_prefixes) for key in new) if expected_prefixes else not new
    retained = {key: tensor for key, tensor in old.items() if key in current}
    assert all(current[key].shape == tensor.shape for key, tensor in retained.items())
    missing = model.load_state_dict(retained, strict=False)
    assert set(missing.missing_keys) == new and not missing.unexpected_keys
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS[arm]
    return model
