# -*- coding: utf-8 -*-
# @FileName: image_encoder_i2v.py
# @Time    : 7/11/24 16:43
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk_image: nn.Module,
        trunk_video: nn.Module,
        neck: nn.Module,
    ):
        super().__init__()
        self.trunk_image = trunk_image
        self.trunk_video = trunk_video
        self.neck = neck

    def forward(self, sample_image: torch.Tensor, sample_video: torch.Tensor):
        # sample_image: (B*T) x C x H x W
        # sample_video: (B*T) x C x self.t_window x H x W

        tinysam_features = self.trunk_image(sample_image)

        tfi_features = self.trunk_video(tinysam_features, sample_video)

        output_features, pos = self.neck(tfi_features)

        output = {
            "backbone_fpn": output_features,
            "vision_pos_enc": pos,
        }

        return output


class FpnNeck(nn.Module):
    """
    add position embedding for the following memory attention
    """
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.d_model = d_model

    def forward(self, xs: List[torch.Tensor]):

        return [xs], [self.position_encoding(xs).to(xs.dtype)]
