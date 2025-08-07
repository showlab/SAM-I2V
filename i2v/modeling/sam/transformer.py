import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from i2v.modeling.position_encoding import apply_rotary_enc, compute_axial_cis, apply_rotary_enc_qk, \
    apply_rotary_enc_qk_st_fusion, apply_rotary_enc_qk_dynamic, apply_rotary_enc_v2
from i2v.modeling.i2v_utils import MLP
from i2v.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
# A fallback setting to allow all available kernels if Flash Attention fails
ALLOW_ALL_KERNELS = False


def sdp_kernel_context(dropout_p):
    """
    Get the context for the attention scaled dot-product kernel. We use Flash Attention
    by default, but fall back to all available kernels if Flash Attention fails.
    """
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        # if Flash attention kernel is off, then math kernel needs to be enabled
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope

        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention_QK(Attention):
    """Attention with rotary position encoding."""
    # allow to set size for query and key/value
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        rope_k_repeat=False,
        q_feat_sizes=(64, 64),
        k_feat_sizes=(32, 32),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        self.freqs_cis_q = self.compute_cis(end_x=q_feat_sizes[0], end_y=q_feat_sizes[1])
        self.freqs_cis_k = self.compute_cis(end_x=k_feat_sizes[0], end_y=k_feat_sizes[1])
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor,
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        self.freqs_cis_q = self.freqs_cis_q.to(q.device)
        self.freqs_cis_k = self.freqs_cis_k.to(k.device)

        q, k = apply_rotary_enc_qk(
            q,
            k,
            freqs_cis_q=self.freqs_cis_q,
            freqs_cis_k=self.freqs_cis_k,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention_Dynamic_Size(Attention):
    """Attention with rotary position encoding."""
    # query size fixed, key/value size is dynamic
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # 默认为64
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        self.freqs_cis_q = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis_k0 = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis_k1 = self.compute_cis(end_x=feat_sizes[0]//2, end_y=feat_sizes[1]//2)
        self.freqs_cis_k2 = self.compute_cis(end_x=feat_sizes[0]//4, end_y=feat_sizes[1]//4)
        self.freqs_cis_k3 = self.compute_cis(end_x=feat_sizes[0]//8, end_y=feat_sizes[1]//8)
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        self.freqs_cis_q = self.freqs_cis_q.to(q.device)

        # 计算 k 的旋转编码
        num_k_rope = k.size(-2) - num_k_exclude_rope
        num_frame = num_k_exclude_rope // 4
        resolution_frame = num_k_rope // num_frame
        assert num_k_exclude_rope % 4 == 0, f"{num_k_exclude_rope} muse be divided by 4"
        assert num_k_rope % num_frame == 0, f"{num_k_rope} must be divided by {num_frame}"

        if num_frame > 1:
            assert self.rope_k_repeat

        if resolution_frame == 64*64:
            self.freqs_cis_k = self.freqs_cis_k0.to(k.device)
        elif resolution_frame == 32*32:
            self.freqs_cis_k = self.freqs_cis_k1.to(k.device)
        elif resolution_frame == 16*16:
            self.freqs_cis_k = self.freqs_cis_k2.to(k.device)
        elif resolution_frame == 8*8:
            self.freqs_cis_k = self.freqs_cis_k3.to(k.device)

        q, k[:, :, :num_k_rope] = apply_rotary_enc_qk_dynamic(
            q,
            k[:, :, :num_k_rope],
            freqs_cis_q=self.freqs_cis_q,
            freqs_cis_k=self.freqs_cis_k,
            repeat_freqs_k=self.rope_k_repeat,
            resolution_frame=resolution_frame,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention_MA(Attention):
    """Masked Attention with rotary position encoding."""
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
            self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # q: (batch_size, seq_len_q, channels_q)
        # k, v: (batch_size, seq_len_kv, channels_kv)

        q = self.q_proj(q)  # (batch_size, seq_len_q, internal_dim)
        k = self.k_proj(k)  # (batch_size, seq_len_kv, internal_dim)
        v = self.v_proj(v)  # (batch_size, seq_len_kv, internal_dim)

        q = self._separate_heads(q, self.num_heads)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = self._separate_heads(k, self.num_heads)  # (batch_size, num_heads, seq_len_kv, head_dim)
        v = self._separate_heads(v, self.num_heads)  # (batch_size, num_heads, seq_len_kv, head_dim)

        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope

        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        if attn_mask is not None:
            # attn_mask's shape is (batch_size, seq_len_kv, 1)
            # convert it to (batch_size, num_heads, seq_len_q, seq_len_kv)

            attn_mask = attn_mask.squeeze(-1)  # (batch_size, seq_len_kv)

            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len_kv)

            attn_mask = attn_mask.expand(-1, self.num_heads, q.size(2), -1)  # (batch_size, num_heads, seq_len_q, seq_len_kv)

        if not attn_mask.any():
            out = q
            print("all masked!")
        else:
            dropout_p = self.dropout_p if self.training else 0.0

            try:
                with sdp_kernel_context(dropout_p):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
                    )
            except Exception as e:
                warnings.warn(
                    f"Flash Attention failed. {e}\n",
                    category=UserWarning,
                    stacklevel=2,
                )
                global ALLOW_ALL_KERNELS
                ALLOW_ALL_KERNELS = True
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
                )

        out = self._recombine_heads(out)  # (batch_size, seq_len_q, internal_dim)
        out = self.out_proj(out)  # (batch_size, seq_len_q, output_dim)

        return out


class RoPEAttention_for_MPG(Attention):
    """Attention with rotary position encoding for memory prompt generator."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        q_num=8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=q_num, end_y=1)
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding

        num_k_rope = k.size(-2) - num_k_exclude_rope

        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis.to(q.device),
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            # Fall back to all kernels if the Flash attention kernel fails
            warnings.warn(
                f"Flash Attention kernel failed due to: {e}\nFalling back to all available "
                f"kernels for scaled_dot_product_attention (which may have a slower speed).",
                category=UserWarning,
                stacklevel=2,
            )
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention_MA_for_MPG(Attention):
    """Masked attention with rotary position encoding for memory prompt generator."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        q_num=3,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        self.freqs_cis_q = self.compute_cis(end_x=q_num, end_y=1)
        self.freqs_cis_k = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])

        self.rope_k_repeat = rope_k_repeat

    def forward(
            self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # q: (batch_size, seq_len_q, channels_q)
        # k, v: (batch_size, seq_len_kv, channels_kv)

        q = self.q_proj(q)  # (batch_size, seq_len_q, internal_dim)
        k = self.k_proj(k)  # (batch_size, seq_len_kv, internal_dim)
        v = self.v_proj(v)  # (batch_size, seq_len_kv, internal_dim)

        q = self._separate_heads(q, self.num_heads)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = self._separate_heads(k, self.num_heads)  # (batch_size, num_heads, seq_len_kv, head_dim)
        v = self._separate_heads(v, self.num_heads)  # (batch_size, num_heads, seq_len_kv, head_dim)

        num_k_rope = k.size(-2) - num_k_exclude_rope

        q, k[:, :, :num_k_rope] = apply_rotary_enc_qk(
            q,
            k[:, :, :num_k_rope],
            freqs_cis_q=self.freqs_cis_q.to(q.device),
            freqs_cis_k=self.freqs_cis_k.to(k.device),
            repeat_freqs_k=self.rope_k_repeat,
        )

        if attn_mask is not None:
            # attn_mask's shape is (batch_size, seq_len_kv, 1)
            # convert it to (batch_size, num_heads, seq_len_q, seq_len_kv)

            attn_mask = attn_mask.squeeze(-1)  # (batch_size, seq_len_kv)

            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len_kv)

            attn_mask = attn_mask.expand(-1, self.num_heads, q.size(2), -1)  # (batch_size, num_heads, seq_len_q, seq_len_kv)

        if not attn_mask.any():
            out = q
            print("all masked!")
        else:
            dropout_p = self.dropout_p if self.training else 0.0

            try:
                with sdp_kernel_context(dropout_p):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
                    )
            except Exception as e:
                warnings.warn(
                    f"Flash Attention failed. {e}\n",
                    category=UserWarning,
                    stacklevel=2,
                )
                global ALLOW_ALL_KERNELS
                ALLOW_ALL_KERNELS = True
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
                )

        out = self._recombine_heads(out)  # (batch_size, seq_len_q, internal_dim)
        out = self.out_proj(out)  # (batch_size, seq_len_q, output_dim)

        return out
