from typing import Sequence

from torch import Tensor, nn

class PatchEmbed(nn.Module):
    patch_size: int

class VisionTransformer(nn.Module):
    embed_dim: int
    num_heads: int
    blocks: nn.ModuleList
    pos_embed: Tensor
    patch_embed: PatchEmbed
    norm: nn.Module

    def __call__(self, x: Tensor, masks: Tensor | None = None) -> Tensor: ...
    def interpolate_pos_encoding(self, x: Tensor, pos_embed: Tensor) -> Tensor: ...
    def eval(self) -> "VisionTransformer": ...

def vit_small(
    img_size: Sequence[int] = ...,
    patch_size: int = ...,
) -> VisionTransformer: ...
