# -*- coding: utf-8 -*-
# @FileName: tfi.py
# @Time    : 7/11/24 16:32
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class IntegrationNetwork(nn.Module):
    def __init__(self, integration_dim=256):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(integration_dim, integration_dim, 3, 1, 1),
            LayerNorm2d(integration_dim),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C1, H, W]

        output = self.layer(x)

        return output


class TemporalNet(nn.Module):
    def __init__(self,
                 temporal_dim=96,
                 temporal_kernel_size=3,
                 temporal_conv_mlp_ratio=1,
                 temporal_loop_layers=3,
                 ):
        super().__init__()
        layers = []
        self.lns = nn.ModuleList()
        for i in range(temporal_loop_layers):
            layers.append(nn.Sequential(OrderedDict([
                (f"c_fc1_{i}", nn.Conv3d(temporal_dim, int(temporal_dim * temporal_conv_mlp_ratio),
                                         kernel_size=(1, 3, 3),
                                         padding=(0, 1, 1))),
                (f"gelu1_{i}", QuickGELU()),
                (f"c_fc2_{i}", nn.Conv3d(int(temporal_dim * temporal_conv_mlp_ratio), temporal_dim,
                                         kernel_size=(temporal_kernel_size, 1, 1),
                                         padding=(temporal_kernel_size // 2, 0, 0))),
            ])))
            self.lns.append(nn.LayerNorm(temporal_dim))
        self.temporal_net = nn.ModuleList(layers)
        self.gelu = QuickGELU()

    def forward(self, x):
        # x: [B, C2, t, H, W]

        out = x
        for i, layer in enumerate(self.temporal_net):
            ln = self.lns[i]
            out = self.gelu(out + layer(ln(out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)))
        return out


class TemporalToSpatialFusion(nn.Module):
    def __init__(self, c1, c2):
        super(TemporalToSpatialFusion, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c1+c2, c1, 1, 1, 0),
            LayerNorm2d(c1),
            nn.GELU()
        )

    def forward(self, spatial_feat, temporal_feat):
        """
        spatial_feat: (b, c1, h, w)
        temporal_feat: (b, c2, t, h, w)
        """

        concat_feat = torch.cat((spatial_feat, temporal_feat[:, :, -1, :, :]), dim=1)

        fused_feat = self.fuse(concat_feat)

        return fused_feat


class SpatialToTemporalFusion(nn.Module):
    def __init__(self, c1, c2):
        super(SpatialToTemporalFusion, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 + c2, c2, 1, 1, 0),
            LayerNorm2d(c2),
            nn.GELU()
        )

    def forward(self, spatial_feat, temporal_feat):
        """
        spatial_feat: (b, c1, h, w)
        temporal_feat: (b, c2, t, h, w)
        """

        concat_feat = torch.cat((spatial_feat, temporal_feat[:, :, -1, :, :]), dim=1)

        fused_feat = self.fuse(concat_feat)

        # create a temporal_feat to avoid in-place modification
        updated_temporal_feat = temporal_feat.clone()
        updated_temporal_feat[:, :, -1, :, :] = fused_feat

        return updated_temporal_feat


class TFI(nn.Module):
    # Temporal Feature Integrator
    def __init__(
            self,
            d_model_list=[160, 320, 256],
            s_patch_size=16,
            t_patch_size=3,
            t_window=5,
            temporal_kernel_size=3,
            temporal_conv_mlp_ratio=1,
            integration_dim=256,
            temporal_dim=96,
            temporal_loop_layers=3,
            selected_layers=None,
            use_checkpoint=False,
    ):
        super().__init__()
        self.selected_layers = selected_layers
        num_layers = len(self.selected_layers)
        spatial_patch_size = s_patch_size
        temporal_patch_size = t_patch_size
        self.t_window = t_window
        self.use_checkpoint = use_checkpoint

        self.temporal_stem = nn.Conv3d(3, temporal_dim,
                                       kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                                       stride=(1, spatial_patch_size, spatial_patch_size),
                                       padding=(temporal_patch_size//2, 0, 0))

        self.input_linears = nn.ModuleList([nn.Conv2d(d_model_list[i], integration_dim, 1, 1, 0) for i in range(num_layers)])
        self.integration2temporal_nets = nn.ModuleList([SpatialToTemporalFusion(integration_dim,
                                                                                temporal_dim,
                                                                                ) for _ in range(num_layers)])
        self.temporal2integration_nets = nn.ModuleList([TemporalToSpatialFusion(integration_dim,
                                                                                temporal_dim,
                                                                                ) for _ in range(num_layers)])
        self.temporal_nets = nn.ModuleList([TemporalNet(temporal_dim,
                                                        temporal_kernel_size,
                                                        temporal_conv_mlp_ratio,
                                                        temporal_loop_layers,
                                                        ) for _ in range(num_layers)])
        self.integration_nets = nn.ModuleList([IntegrationNetwork(integration_dim,
                                                                  ) for _ in range(num_layers)])

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'skip_init') and m.skip_init:
            return
        from timm.models.layers import trunc_normal_
        init_type = 'trunc_normal_'
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            if init_type == 'trunc_normal_':
                trunc_normal_(m.weight, std=.02)
            elif init_type == 'xavier_uniform_':
                nn.init.xavier_uniform_(m.weight)
            else:
                raise NotImplementedError
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_spatial_features, input_video):
        for idx, tensor in enumerate(input_spatial_features):
            B, C, H, W = tensor.shape
            if H != 64 or W != 64:
                tensor_resized = F.interpolate(tensor, size=(64, 64), mode='bilinear', align_corners=False)
                input_spatial_features[idx] = tensor_resized

        x_temporal = self.temporal_stem(input_video)  # [B*num_frames, C2, t_window, 64, 64]
        res_feat = 0.0

        outputs = []
        for idx, layer_id in enumerate(self.selected_layers):
            x_temporal = self.temporal_nets[idx](x_temporal)  # [B*num_frames, C2, t_window, 64, 64]

            mid_feat = self.input_linears[idx](input_spatial_features[layer_id]) + res_feat  # [B*num_frames, C1, H, W]

            updated_x_temporal = self.integration2temporal_nets[idx](mid_feat, x_temporal)  # [B*num_frames, C2, t_window, H, W]

            updated_mid_feat = self.temporal2integration_nets[idx](mid_feat, x_temporal)  # [B*num_frames, C1, H, W]

            res_feat = self.integration_nets[idx](updated_mid_feat)  # [B, num_frames, C1, H, W]

            x_temporal = updated_x_temporal  # [B*num_frames, C2, t_window, H, W]

            outputs.append(res_feat + updated_mid_feat)  # 3 x [B*num_frames, C1, H, W]

        return outputs[-1]
