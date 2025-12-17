import math

import torch
from torch import nn

from avp_vit.model.hidden import HiddenStreamParams


def test_init_hidden_shape():
    dim, n_reg = 64, 4
    params = HiddenStreamParams(dim, n_reg, use_recurrence_ln=False)
    hidden = params.init_hidden(batch_size=2, n_spatial=16)
    assert hidden.shape == (2, 1 + n_reg + 16, dim)


def test_init_hidden_no_registers():
    dim = 64
    params = HiddenStreamParams(dim, n_registers=0, use_recurrence_ln=False)
    hidden = params.init_hidden(batch_size=2, n_spatial=16)
    assert hidden.shape == (2, 1 + 16, dim)


def test_normalize_shape():
    dim, n_reg = 64, 4
    params = HiddenStreamParams(dim, n_reg, use_recurrence_ln=True)
    hidden = torch.randn(2, 1 + n_reg + 16, dim)
    out = params.normalize(hidden)
    assert out.shape == hidden.shape


def test_normalize_no_registers():
    dim = 64
    params = HiddenStreamParams(dim, n_registers=0, use_recurrence_ln=True)
    hidden = torch.randn(2, 1 + 16, dim)
    out = params.normalize(hidden)
    assert out.shape == hidden.shape


def test_recurrence_ln_weight_init():
    dim = 64
    params = HiddenStreamParams(dim, n_registers=4, use_recurrence_ln=True)
    expected = 1.0 / math.sqrt(dim)
    assert isinstance(params.cls_ln, nn.LayerNorm)
    assert isinstance(params.reg_ln, nn.LayerNorm)
    assert isinstance(params.spatial_ln, nn.LayerNorm)
    assert torch.allclose(params.cls_ln.weight, torch.full((dim,), expected))
    assert torch.allclose(params.reg_ln.weight, torch.full((dim,), expected))
    assert torch.allclose(params.spatial_ln.weight, torch.full((dim,), expected))


def test_recurrence_ln_disabled():
    dim = 64
    params = HiddenStreamParams(dim, n_registers=4, use_recurrence_ln=False)
    assert isinstance(params.cls_ln, nn.Identity)
    assert isinstance(params.reg_ln, nn.Identity)
    assert isinstance(params.spatial_ln, nn.Identity)


def test_n_prefix():
    params = HiddenStreamParams(64, n_registers=8, use_recurrence_ln=False)
    assert params.n_prefix == 1 + 8
