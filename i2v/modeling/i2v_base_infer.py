# -*- coding: utf-8 -*-
# @FileName: i2v_base_infer.py
# @Time    : 10/11/24 20:23
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

import torch
import random
import numpy as np
import heapq

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

import torch.distributed
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from i2v.modeling.sam.mask_decoder_mpg import MaskDecoderMPG  # mask decoder compatible with memory prompt generator
from i2v.modeling.sam.prompt_encoder import PromptEncoder
from i2v.modeling.sam.memory_prompt_generator_fg import MemoryPromptGeneratorFG  # foreground memory prompt generator

from i2v.modeling.sam.transformer import TwoWayTransformer
from i2v.modeling.i2v_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class I2VBase(torch.nn.Module):
    def __init__(
            self,
            image_encoder,
            memory_attention,
            memory_encoder,
            memory_prompt_generator,
            num_maskmem=7,  # default 1 input frame + 6 previous frames
            num_maskmem_global_all=80,
            num_maskmem_global=4,
            num_maskmem_local_all=8,
            num_maskmem_local=2,
            iou_threshold=0.5,
            area_threshold=100,
            image_size=512,
            backbone_stride=16,  # stride of the image backbone output
            sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
            sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
            # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
            binarize_mask_from_pts_for_mem_enc=False,
            use_mask_input_as_output_without_sam=False,
            # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
            # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
            # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
            # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
            max_cond_frames_in_attn=-1,
            # on the first frame, whether to directly add the no-memory embedding to the image feature
            # (instead of using the transformer encoder)
            directly_add_no_mem_embed=False,
            # whether to use high-resolution feature maps in the SAM mask decoder
            use_high_res_features_in_sam=False,
            # whether to output multiple (3) masks for the first click on initial conditioning frames
            multimask_output_in_sam=False,
            # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
            # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
            multimask_min_pt_num=1,
            multimask_max_pt_num=1,
            # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
            multimask_output_for_tracking=False,
            # Whether to use multimask tokens for obj ptr; Only relevant when both
            # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
            use_multimask_token_for_obj_ptr: bool = False,
            # whether to use sigmoid to restrict ious prediction to [0-1]
            iou_prediction_use_sigmoid=False,
            # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
            # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
            # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
            memory_temporal_stride_for_eval=1,
            # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
            non_overlap_masks_for_mem_enc=False,
            # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
            use_obj_ptrs_in_encoder=False,
            # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
            max_obj_ptrs_in_encoder=16,
            # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
            add_tpos_enc_to_obj_ptrs=True,
            # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
            # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
            proj_tpos_enc_in_obj_ptrs=False,
            # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
            # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
            use_signed_tpos_enc_to_obj_ptrs=False,
            # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
            # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
            only_obj_ptrs_in_the_past_for_eval=False,
            # Whether to predict if there is an object in the frame
            pred_obj_scores: bool = False,
            # Whether to use an MLP to predict object scores
            pred_obj_scores_mlp: bool = False,
            # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
            # Whether to have a fixed no obj pointer when there is no object present
            # or to use it as an additive embedding with obj_ptr produced by decoder
            fixed_no_obj_ptr: bool = False,
            # Soft no object, i.e. mix in no_obj_ptr softly,
            # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
            soft_no_obj_ptr: bool = False,
            use_mlp_for_obj_ptr_proj: bool = False,
            # add no obj embedding to spatial frames
            no_obj_embed_spatial: bool = False,
            # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
            sam_mask_decoder_extra_args=None,
            compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam

        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1

        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
                self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        self.num_maskmem_global_all = num_maskmem_global_all
        self.num_maskmem_global = num_maskmem_global
        self.num_maskmem_local_all = num_maskmem_local_all
        self.num_maskmem_local = num_maskmem_local
        self.iou_threshold = iou_threshold
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # memory_prompt_generator
        self.memory_prompt_generator = memory_prompt_generator

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in I2VVideoPredictor for inference or I2VTrain for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

        # mask decoder compatible with memory prompt generator
        self.sam_mask_decoder = MaskDecoderMPG(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
            self,
            backbone_features,
            point_inputs=None,
            mask_inputs=None,
            memory=None,
            memory_pos_embed=None,
            mask_fg=None,
            mask_bg=None,
            num_obj_ptr_tokens=0,
            high_res_features=None,
            multimask_output=False,
    ):
        """
        Forward SAM prompt encoders, memory prompt generator, and mask decoder.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - memory: [N, B, C]
        - memory_pos_embed: [N, B, C]
        - mask_fg: [N, B, 1]
        - mask_bg: [N, B, 1]
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # 0. Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # 1. Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        # 2. prompt encoder
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )

        # 3. memory prompt generator
        memory_prompt_embeddings_fg = self.memory_prompt_generator(
            memory=memory,
            memory_pos_embed=memory_pos_embed,
            mask_fg=mask_fg,
            mask_bg=mask_bg,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
            bs=B,
        )

        # 4. mask decoder
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            memory_prompt_embeddings_fg=memory_prompt_embeddings_fg,  # foreground_memory as a prompt
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # 5. Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)

        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into an output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            memory = None
            memory_pos_embed = None
            mask_fg = None
            mask_bg = None
            num_obj_ptr_tokens = 0

            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                memory=memory,
                memory_pos_embed=memory_pos_embed,
                mask_fg=mask_fg,
                mask_bg=mask_bg,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
                high_res_features=high_res_features,
            )

        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)

        return backbone_out

    def forward_image_video(self, img_batch: torch.Tensor, vid_batch: torch.Tensor):
        # img_batch: (B*T) x C x H x W
        # vid_batch: (B*T) x C x self.t_window x H x W

        backbone_out = self.image_encoder(img_batch, vid_batch)

        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()

        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            output_dict,
            num_frames,
            track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        # current_vision_feats is a list, in which the element shape is (HW)BC
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            memory = None
            memory_pos_embed = None
            mask_fg = None
            mask_bg = None
            num_obj_ptr_tokens = 0
            return pix_feat, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens

        ########################################
        ######## Start Memory Selection ########
        ########################################

        # Step 1: construct the memory used to condition the visual features of the current frame
        if not is_init_cond_frame:
            # Initialize memory features and positional encodings for each batch.
            batch_memory_list = []
            batch_mask_fg_list = []
            batch_mask_bg_list = []
            batch_to_cat_memory_pos_embed_list = []
            # for save object pointers info
            batch_num_obj_ptr_tokens_list = []

            # process each object in the batch
            for b in range(B):
                # Retrieve the memories encoded with the maskmem backbone
                to_cat_memory, to_cat_memory_pos_embed = [], []
                to_cat_mask_fg = []
                to_cat_mask_bg = []
                assert len(output_dict["cond_frame_outputs"]) > 0
                # Select a maximum number of temporally closest cond frames for cross attention
                cond_outputs = output_dict["cond_frame_outputs"]
                selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                    frame_idx, cond_outputs, self.max_cond_frames_in_attn
                )
                # Set the positions of prompt frames to 0
                t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]  # `out` is a large dictionary that contains all the information

                # 0. Define a list to store the selected frames for later adding the corresponding object pointers
                global_selected_idxs = []
                local_selected_idxs = []
                fill_idxs = []

                # 1. Identify all non-prompt frames
                non_cond_outputs = output_dict["non_cond_frame_outputs"]

                non_cond_frame_idxs = [int(x) for x in non_cond_outputs.keys()]
                non_cond_frame_idxs.sort()  # Sort in ascending order.

                # 2. global
                # 2.1 Retrieve the global frames
                global_all_idxs = [
                    x for x in non_cond_frame_idxs
                    if x < frame_idx - self.num_maskmem_local_all
                       and x >= frame_idx - self.num_maskmem_local_all - 1 * self.num_maskmem_global_all
                ]

                # 2.2 Filter based on similarity
                if len(global_all_idxs) > 0:
                    # (1) Extract all `image_embed` corresponding to the selected indices
                    image_embeds = [non_cond_outputs[idx]['image_embed'][:, b:b + 1, :].to(device, non_blocking=True)
                                    for idx in global_all_idxs]  # List of [N, B=1, C]

                    # (2) Stack them into a tensor with shape `[M, N, B=1, C]`, where `M` is the number of selected indices
                    image_embeds = torch.stack(image_embeds, dim=0)  # [M, N, B=1, C]

                    # (3) Retrieve the features from the last scale of `current_vision_feats`, with shape `[N, B=1, C]`
                    current_feat = current_vision_feats[-1][:, b:b + 1, :]  # [N, B=1, C]

                    assert image_embeds.size(2) == 1, "Batch size must be 1"
                    assert current_feat.size(1) == 1, "Batch size must be 1"

                    # (4) Remove the batch dimension and reshape the tensors to `[M, N, C]` and `[N, C]`
                    image_embeds = image_embeds.squeeze(2)  # [M, N, C]
                    current_feat = current_feat.squeeze(1)  # [N, C]

                    # (5) Flatten the feature dimension by reshaping `[N, C]` to `[N*C]`
                    M, N, C = image_embeds.size()
                    image_embeds_flat = image_embeds.reshape(M, N * C)  # [M, N*C]
                    current_feat_flat = current_feat.reshape(N * C)  # [N*C]

                    # (6) Perform L2 normalization
                    image_embeds_norm = F.normalize(image_embeds_flat, p=2, dim=1)  # [M, N*C]
                    current_feat_norm = F.normalize(current_feat_flat, p=2, dim=0)  # [N*C]

                    # (7) Compute similarity using `torch.mm` for matrix multiplication, resulting in a tensor of shape `[M]`
                    similarity = torch.mm(image_embeds_norm, current_feat_norm.unsqueeze(1)).squeeze(1)  # [M]

                    # (8) Apply softmax to the similarity scores
                    probs = F.softmax(similarity, dim=0)  # [M]

                    # (9) Determine the number of samples
                    num_samples = min(len(global_all_idxs), self.num_maskmem_global)

                    # (10) Use `multinomial` for sampling without replacement
                    sampled_indices = torch.multinomial(probs, num_samples, replacement=False)  # [num_samples]

                    # (11) Retrieve the indices after sampling
                    sampled_idxs = [global_all_idxs[idx.item()] for idx in sampled_indices]
                    sampled_idxs.sort()  # Sort in ascending order

                    # (12) Add the sampled indices and their corresponding `non_cond_outputs` to the result list, assigning them incrementing integer IDs
                    for i, idx in enumerate(sampled_idxs, 1):
                        t_pos_and_prevs.append((i, non_cond_outputs[idx]))
                        global_selected_idxs.append(idx)

                # 3. local
                # 3.1 Retrieve the local frames
                local_all_idxs = [x for x in non_cond_frame_idxs if x >= frame_idx - self.num_maskmem_local_all and x < frame_idx]

                # 3.2 Filter based on similarity
                if len(local_all_idxs) > 0:
                    # (1) Extract all `image_embed` corresponding to the selected indices
                    image_embeds = [non_cond_outputs[idx]['image_embed'][:, b:b + 1, :].to(device, non_blocking=True)
                                    for idx in
                                    local_all_idxs]  # List of [N, B=1, C]

                    # (2) Stack them into a tensor with shape `[M, N, B=1, C]`, where `M` is the number of selected indices
                    image_embeds = torch.stack(image_embeds, dim=0)  # [M, N, B=1, C]

                    # (3) Retrieve the features from the last scale of `current_vision_feats`, with shape `[N, B=1, C]`
                    current_feat = current_vision_feats[-1][:, b:b + 1, :]  # [N, B=1, C]

                    assert image_embeds.size(2) == 1, "Batch size must be 1"
                    assert current_feat.size(1) == 1, "Batch size must be 1"

                    # (4) Remove the batch dimension and reshape the tensors to `[M, N, C]` and `[N, C]`
                    image_embeds = image_embeds.squeeze(2)  # [M, N, C]
                    current_feat = current_feat.squeeze(1)  # [N, C]

                    # (5) Flatten the feature dimension by reshaping `[N, C]` to `[N*C]`
                    M, N, C = image_embeds.size()
                    image_embeds_flat = image_embeds.reshape(M, N * C)  # [M, N*C]
                    current_feat_flat = current_feat.reshape(N * C)  # [N*C]

                    # (6) Perform L2 normalization
                    image_embeds_norm = F.normalize(image_embeds_flat, p=2, dim=1)  # [M, N*C]
                    current_feat_norm = F.normalize(current_feat_flat, p=2, dim=0)  # [N*C]

                    # (7) Compute similarity using `torch.mm` for matrix multiplication, resulting in a tensor of shape `[M]`
                    similarity = torch.mm(image_embeds_norm, current_feat_norm.unsqueeze(1)).squeeze(1)  # [M]

                    # (8) Apply softmax to the similarity scores
                    probs = F.softmax(similarity, dim=0)  # [M]

                    # (9) Determine the number of samples
                    num_samples = min(len(local_all_idxs), self.num_maskmem_local)

                    # (10) Use `multinomial` for sampling without replacement
                    sampled_indices = torch.multinomial(probs, num_samples, replacement=False)  # [num_samples]

                    # (11) Retrieve the indices after sampling
                    sampled_idxs = [local_all_idxs[idx.item()] for idx in sampled_indices]
                    sampled_idxs.sort()  # Sort in ascending order

                    # (12) Add the sampled indices and their corresponding `non_cond_outputs` to the result list, assigning them incrementing integer IDs
                    for i, idx in enumerate(sampled_idxs, self.num_maskmem_global + 1):
                        t_pos_and_prevs.append((i, non_cond_outputs[idx]))
                        local_selected_idxs.append(idx)

                # 4. Pad to a fixed length to allow stacking along the batch dimension
                count_i_zero = sum(1 for i, out in t_pos_and_prevs if i == 0)  # The number of prompt frames, assumed to be the same across all batches
                count_i_not_zero = sum(1 for i, out in t_pos_and_prevs if i != 0)  # The number of selected non-prompt frames, which may vary across different batches
                num_maskmem_global_local = self.num_maskmem_global + self.num_maskmem_local
                fixed_non_cond_memory_length = frame_idx - 1 if frame_idx <= num_maskmem_global_local else num_maskmem_global_local
                if count_i_not_zero < fixed_non_cond_memory_length:
                    fill_num = fixed_non_cond_memory_length - count_i_not_zero
                    fill_content = t_pos_and_prevs[0]
                    for _ in range(fill_num):
                        t_pos_and_prevs.insert(0, fill_content)
                        fill_idxs.insert(0, 0)
                    assert len(t_pos_and_prevs) == fixed_non_cond_memory_length + count_i_zero

                # 5. Merge all the selected memory features together
                for t_pos, prev in t_pos_and_prevs:
                    if prev is None:
                        continue  # skip padding frames

                    # (1) Process the `feats`
                    # "maskmem_features" might have been offloaded to CPU in demo use cases,
                    # so we load it back to GPU (it's a no-op if it's already on GPU).
                    feats = prev["maskmem_features"][b:b + 1, :, :, :].to(device, non_blocking=True)
                    to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))

                    # (2) Process the `mask`
                    masks = prev["pred_masks"][b:b + 1, :, :, :].to(device, non_blocking=True)  # [1, 1, H2=1/4, W2=1/4]
                    masks = torch.sigmoid(masks)  # [1, 1, H2, W2]
                    masks_resized = F.interpolate(masks, size=(feats.shape[2], feats.shape[3]), mode='bilinear', align_corners=False)  # [1, 1, H1, W1]
                    bin_masks_fg = (masks_resized >= 0.5)  # [1, 1, H1, W1], boolean type, where `False` indicates masking (i.e., the position is masked out)
                    bin_masks_bg = (masks_resized < 0.5)  # [1, 1, H1, W1], boolean type, where `False` indicates masking (i.e., the position is masked out)
                    bin_masks_fg = bin_masks_fg.flatten(2).permute(2, 0, 1)  # [H1xW1, B=1, 1]
                    bin_masks_bg = bin_masks_bg.flatten(2).permute(2, 0, 1)  # [H1xW1, B=1, 1]
                    to_cat_mask_fg.append(bin_masks_fg)
                    to_cat_mask_bg.append(bin_masks_bg)

                    # (3) Process positional encodings. It might have been offloaded to CPU in eval
                    maskmem_enc = prev["maskmem_pos_enc"][-1][b:b + 1, :, :, :].to(device)
                    maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                    maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1])  # Temporal positional encoding
                    to_cat_memory_pos_embed.append(maskmem_enc)

                # 6. Add object pointers. Here, "encoder" refers to the memory encoder
                # Also add to `to_cat_mask_fg/bg` to match the number of features and masks
                # `mask_fg` is always True (do not mask object pointers)
                # `mask_bg` is always False (mask out object pointers)
                if self.use_obj_ptrs_in_encoder:
                    # 6.1 First add those object pointers from selected conditioning frames
                    # (optionally, only include object pointers in the past during evaluation)
                    if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                        ptr_cond_outputs = {
                            t: out
                            for t, out in selected_cond_outputs.items()
                            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                        }
                    else:
                        ptr_cond_outputs = selected_cond_outputs
                    # We use fixed positional encoding based on distance from the current frame, with prompt frames being the farthest (i.e., 7)
                    pos_and_ptrs = [
                        (
                            fixed_non_cond_memory_length + 1, out["obj_ptr"][b:b + 1, :],  # [B=1, C]
                        )
                        for t, out in ptr_cond_outputs.items()
                    ]

                    # 6.2 Then add the object pointers for the non-prompt frames
                    # Global frames have medium distance, i.e., 6, 5, 4, 3
                    for i, idx in enumerate(global_selected_idxs, 1):
                        pos_and_ptrs.append(
                            (fixed_non_cond_memory_length + 1 - i, non_cond_outputs[idx]["obj_ptr"][b:b + 1, :])
                        )
                    # Local frames are the closest, i.e., 2, 1
                    for i, idx in enumerate(local_selected_idxs, self.num_maskmem_global + 1):
                        pos_and_ptrs.append(
                            (fixed_non_cond_memory_length + 1 - i, non_cond_outputs[idx]["obj_ptr"][b:b + 1, :])
                        )

                    # 6.3 We need to ensure that the length of object pointers is the same for each batch
                    while len(pos_and_ptrs) < len(t_pos_and_prevs):
                        fill_ptrs_content = pos_and_ptrs[0]
                        pos_and_ptrs.insert(0, fill_ptrs_content)

                    # 6.4 Add the selected object pointers into the memory
                    if len(pos_and_ptrs) > 0:
                        pos_list, ptrs_list = zip(*pos_and_ptrs)
                        # Stack object pointers along dim=0 into [ptr_seq_len, B=1, C] shape
                        obj_ptrs = torch.stack(ptrs_list, dim=0)
                        # Temporal positions: 0 for prompt frames, 1–4 for global frames, and 5–6 for local frames
                        # (sine embedding normalized by the max pointer num).
                        if self.add_tpos_enc_to_obj_ptrs:
                            t_diff_max = fixed_non_cond_memory_length + 1
                            tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                            obj_pos = torch.tensor(pos_list, device=device)
                            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                            obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                            obj_pos = obj_pos.unsqueeze(1).expand(-1, 1, self.mem_dim)  # The batch dimension is 1
                        else:
                            obj_pos = obj_ptrs.new_zeros(len(pos_list), 1, self.mem_dim)  # The batch dimension is 1

                        if self.mem_dim < C:
                            # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                            obj_ptrs = obj_ptrs.reshape(
                                -1, 1, C // self.mem_dim, self.mem_dim
                            )
                            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                            obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                        to_cat_memory.append(obj_ptrs)
                        to_cat_memory_pos_embed.append(obj_pos)

                        # Add the corresponding number of masks.
                        bin_masks_fg_op = torch.ones(obj_ptrs.size(0), obj_ptrs.size(1), 1, dtype=torch.bool).to(device, non_blocking=True)
                        bin_masks_bg_op = torch.zeros(obj_ptrs.size(0), obj_ptrs.size(1), 1, dtype=torch.bool).to(device, non_blocking=True)
                        to_cat_mask_fg.append(bin_masks_fg_op)
                        to_cat_mask_bg.append(bin_masks_bg_op)
                        num_obj_ptr_tokens_per_batch = obj_ptrs.shape[0]
                    else:
                        num_obj_ptr_tokens_per_batch = 0

                memory_per_batch = torch.cat(to_cat_memory, dim=0)  # list of [n, 1, C] concatenate into [N, 1, C]
                memory_pos_embed_per_batch = torch.cat(to_cat_memory_pos_embed, dim=0)  # list of [n, 1, C] concatenate into [N, 1, C]
                mask_fg_per_batch = torch.cat(to_cat_mask_fg, dim=0)  # list of [n, 1, C=1] concatenate into [N, 1, C=1]
                mask_bg_per_batch = torch.cat(to_cat_mask_bg, dim=0)  # list of [n, 1, C=1] concatenate into [N, 1, C=1]

                batch_memory_list.append(memory_per_batch)  # Each addition has the shape [N, 1, C]
                batch_to_cat_memory_pos_embed_list.append(memory_pos_embed_per_batch)  # Each addition has the shape [N, 1, C]
                batch_mask_fg_list.append(mask_fg_per_batch)  # Each addition has the shape [N, 1, C=1]
                batch_mask_bg_list.append(mask_bg_per_batch)  # Each addition has the shape [N, 1, C=1]

                batch_num_obj_ptr_tokens_list.append(num_obj_ptr_tokens_per_batch)
        else:
            # Initial prompt frame for point/bbox (mask prompt frames won’t enter this function)
            # Set memory-related fields to None or 0, as prompt frames aren’t used in the correction process
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                memory = None
                memory_pos_embed = None
                mask_fg = None
                mask_bg = None
                num_obj_ptr_tokens = 0
                return pix_feat_with_mem, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(batch_memory_list, dim=1)  # Concatenate the list of [N, 1, C] tensors into a single [N, B, C] tensor
        memory_pos_embed = torch.cat(batch_to_cat_memory_pos_embed_list, dim=1)  # Concatenate the list of [N, 1, C] tensors into a single [N, B, C] tensor
        mask_fg = torch.cat(batch_mask_fg_list, dim=1)  # Concatenate the list of [N, 1, C=1] tensors into a single [N, B, C=1] tensor
        mask_bg = torch.cat(batch_mask_bg_list, dim=1)  # Concatenate the list of [N, 1, C=1] tensors into a single [N, B, C=1] tensor
        assert all(x == batch_num_obj_ptr_tokens_list[0] for x in batch_num_obj_ptr_tokens_list), "The elements in the list are not all equal"
        num_obj_ptr_tokens = batch_num_obj_ptr_tokens_list[0]

        ######################################
        ######## End Memory Selection ########
        ######################################

        ########################################
        ######## Start Memory Attention ########
        ########################################
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        ######################################
        ######## End Memory Attention ########
        ######################################

        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).contiguous().view(B, C, H, W)

        return pix_feat_with_mem, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens

    def _encode_new_memory(
            self,
            current_vision_feats,
            feat_sizes,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # Memory Encoder
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )

        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                                        1 - is_obj_appearing[..., None, None]
                                ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # We return `image_embed` to compute similarity during memory selection
        image_embed = current_vision_feats[-1]  # (HW)BC

        # If the first frame provides a mask, output it directly
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])

            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )

            memory = None
            memory_pos_embed = None
            mask_fg = None
            mask_bg = None
            num_obj_ptr_tokens = 0

        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)  # return bool value
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                memory=memory,
                memory_pos_embed=memory_pos_embed,
                mask_fg=mask_fg,
                mask_bg=mask_bg,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        # Return memory-related variables because if correction is needed on the current frame, the memory information must be passed to the mask decoder during correction
        return current_out, sam_outputs, high_res_features, pix_feat, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens, image_embed

    def _encode_memory_in_output(
            self,
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse=False,  # tracking in reverse time order (for demo usage)
            # Whether to run the memory encoder on the predicted masks. Sometimes we might want
            # to skip the memory encoder with `run_mem_encoder=False`. For example,
            # in demo we might call `track_step` multiple times for each user click,
            # and only encode the memory when the user finalizes their clicks. And in ablation
            # settings like SAM training on static images, we don't need the memory encoder.
            run_mem_encoder=True,
            # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
            prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, high_res_features, pix_feat, memory, memory_pos_embed, mask_fg, mask_bg, num_obj_ptr_tokens, image_embed = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # for memory selection
        current_out["image_embed"] = image_embed
        current_out["ious"] = ious

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
                self.multimask_output_in_sam
                and (is_init_cond_frame or self.multimask_output_for_tracking)
                and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
