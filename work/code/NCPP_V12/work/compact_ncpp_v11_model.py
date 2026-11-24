"""Same shared prior and parameter tensors; only bottleneck sampling changes."""
import torch
from compact_ncpp_v10_model import StageConditionedNCPP

ARMS = ['continue48', 'dilate_mild48', 'dilate_wide48']
DILATIONS = {'continue48': (1, 1, 1, 1),
             'dilate_mild48': (1, 2, 1, 3),
             'dilate_wide48': (1, 2, 3, 5)}


def model_for(arm):
    assert arm in ARMS
    model = StageConditionedNCPP(stages=4, width=48)
    layers = [model.prior.down2[block].net[index] for block in [2, 3] for index in [0, 2]]
    for layer, dilation in zip(layers, DILATIONS[arm]):
        assert layer.kernel_size == (3, 3) and layer.stride == (1, 1)
        layer.dilation = (dilation, dilation)
        layer.padding = (dilation, dilation)
    return model


def initialize(arm, path):
    model = model_for(arm)
    # No new weights or branches. Changed dilation is recorded in config, not
    # state_dict: candidate forward functions intentionally differ at transfer.
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state, strict=True)
    return model
