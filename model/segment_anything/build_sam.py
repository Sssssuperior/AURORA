# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch

from .modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam,
                       TwoWayTransformer)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

import torch
import torch.nn.functional as F

def interpolate_pos_encoding(state_dict, model):
    """
    Interpolate pre-trained position encodings to match the new size.
    """
    if "image_encoder.pos_embed" in state_dict:
        pretrained_pos_embed = state_dict["image_encoder.pos_embed"] 
        new_pos_embed = model.image_encoder.pos_embed  

        orig_size = pretrained_pos_embed.shape[1:3]
        new_size = new_pos_embed.shape[1:3]

        new_pos_embed_interp = F.interpolate(
            pretrained_pos_embed.permute(0, 3, 1, 2),  # [1, 1280, 64, 64]
            size=new_size,  # (14, 14)
            mode="bicubic",
            align_corners=False
        ).permute(0, 2, 3, 1)  # [1, 14, 14, 1280]

        state_dict["image_encoder.pos_embed"] = new_pos_embed_interp

    return state_dict


def interpolate_rel_pos(state_dict, model):
    """
    Interpolates relative position encodings.
    """
    for i in [7, 15, 23, 31]:  
        for axis in ["h", "w"]:
            key = f"image_encoder.blocks.{i}.attn.rel_pos_{axis}"
            if key in state_dict:
                pretrained_rel_pos = state_dict[key]  
                new_rel_pos = model.state_dict()[key] 

                orig_size = pretrained_rel_pos.shape[0]
                new_size = new_rel_pos.shape[0]
                
                new_rel_pos_interp = F.interpolate(
                    pretrained_rel_pos.unsqueeze(0).unsqueeze(0),  # [1, 1, 127, 80]
                    size=(new_size, 80),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [27, 80]

                state_dict[key] = new_rel_pos_interp

    return state_dict



def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        state_dict = interpolate_pos_encoding(state_dict, sam)
        state_dict = interpolate_rel_pos(state_dict, sam)
        sam.load_state_dict(state_dict, strict=False)
    return sam
