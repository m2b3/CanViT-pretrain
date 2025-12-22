from torch import Tensor, dtype, nn

class RopePositionEncoding:
    periods: Tensor
    dtype: dtype

class DinoVisionTransformer(nn.Module):
    embed_dim: int
    num_heads: int
    n_storage_tokens: int
    patch_size: int
    blocks: nn.ModuleList
    rope_embed: RopePositionEncoding
    norm: nn.Module

    def init_weights(self) -> None: ...
    def forward_features(self, x: Tensor) -> dict[str, Tensor]: ...
    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Tensor | None
    ) -> tuple[Tensor, tuple[int, int]]: ...

def vit_small(
    img_size: int = ...,
    patch_size: int = ...,
    pos_embed_rope_dtype: str = ...,
) -> DinoVisionTransformer: ...
