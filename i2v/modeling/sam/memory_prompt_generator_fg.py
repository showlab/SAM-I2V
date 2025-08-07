# -*- coding: utf-8 -*-
# @FileName: memory_prompt_generator_fg.py
# @Time    : 5/11/24 01:41
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

import torch
import torch.nn as nn

from i2v.modeling.sam.transformer import RoPEAttention, RoPEAttention_MA_for_MPG
from i2v.modeling.i2v_utils import get_activation_fn, get_clones


class MemoryPromptGeneratorFG(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_queries,
        activation: str,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
    ):
        super(MemoryPromptGeneratorFG, self).__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.self_attn = self_attention
        self.cross_attn = cross_attention

        self.no_memory_prompt_embeddings_fg = nn.Embedding(1, embed_dim)

        self.query_embedding_fg = nn.Embedding(num_queries, embed_dim)

        self.norm_fg_sa = nn.LayerNorm(embed_dim)
        self.dropout_fg_sa = nn.Dropout(dropout)
        self.norm_fg_ca = nn.LayerNorm(embed_dim)
        self.dropout_fg_ca = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.activation1 = get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def forward(self, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens=0, bs=1):
        """
        generate foreground prompt

        Inputs:
        - memory: (N1, B, C1)
        - memory_pos_embed: (N1, B, C1)
        - mask_fg: (N1, B, C2)

        Outputs:
        - memory_prompt_embeddings_fg: (B, N2, C2)
        """

        if memory is None:
            memory_prompt_embeddings_fg = self.no_memory_prompt_embeddings_fg.weight.expand(bs, self.num_queries, -1)
            return memory_prompt_embeddings_fg

        # NBC -> BNC
        B = memory.size(1)
        memory = memory.transpose(0, 1)
        memory_pos_embed = memory_pos_embed.transpose(0, 1)
        mask_fg = mask_fg.transpose(0, 1)

        # 1
        queries_fg = self.query_embedding_fg.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N2, C2)

        # 2
        queries_fg_ca = self._forward_fg_ca(queries_fg, memory, query_pos=None, pos=memory_pos_embed, attn_mask=mask_fg, num_k_exclude_rope=num_obj_ptr_tokens)

        # 3
        queries_fg_sa = self._forward_fg_sa(queries_fg_ca, query_pos=None)

        # 4
        fg_residual = self.norm1(queries_fg_sa)
        fg_residual = self.linear2(self.dropout1(self.activation1(self.linear1(fg_residual))))
        memory_prompt_embeddings_fg = queries_fg_sa + self.dropout2(fg_residual)

        return memory_prompt_embeddings_fg

    def _forward_fg_sa(self, tgt, query_pos):
        """
        Inputs:
        - tgt: (B, N, C)
        - query_pos: (1, N, C)

        Outputs:
        - tgt: (B, N, C)
        """
        # Self-Attention
        tgt2 = self.norm_fg_sa(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout_fg_sa(tgt2)
        return tgt

    def _forward_fg_ca(self, tgt, memory, query_pos, pos, attn_mask, num_k_exclude_rope=0):
        """
        Inputs:
        - tgt: (B, N2, C2)
        - memory: (B, N1, C1)
        - query_pos: (B, N2, C2)
        - pos: (B, N1, C1)
        - attn_mask: mask for masked cross attention (B, N2, C2)
        - num_k_exclude_rope: optional parameter

        Outputs:
        - tgt: (B, N2, C2)
        """
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn, RoPEAttention_MA_for_MPG)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm_fg_ca(tgt)
        q = tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2
        k = memory + pos if self.pos_enc_at_cross_attn_keys else memory
        tgt2 = self.cross_attn(q=q, k=k, v=memory, attn_mask=attn_mask, **kwds)
        tgt = tgt + self.dropout_fg_ca(tgt2)
        return tgt
