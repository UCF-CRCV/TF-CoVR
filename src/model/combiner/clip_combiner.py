"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

# from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather
from collections import defaultdict
from src.model.blip.loss import FalseNegativeContrastiveLoss
import clip
from src.model.combiner.combiner import Combiner, CrossAttentionCombiner


def build_false_negative_dict(pairs, N):
    """
    Converts a list of (i, f) pairs into a dict: i -> set of false negative indices
    """

    fn_dict = defaultdict(set)
    idx_dict = defaultdict(set)
    for idx, (i, f) in enumerate(pairs):
        fn_dict[i].add(idx)

    return fn_dict


class CLIP2Cir(nn.Module):
    """
    CLIP first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="clip_L",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,
        si_el_weight=0,
        si_att_weight=0,
        **kwargs
    ):
        super().__init__()

        self.loss_fn_elimination = FalseNegativeContrastiveLoss(mode="elimination")
        self.loss_fn_attraction = FalseNegativeContrastiveLoss(mode="attraction")
        self.loss = loss

        # self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # self.train_vit = train_vit
        # if not train_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features, cross_attention_freq
        # )
        # self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # state_dict = self.Qformer.state_dict()
        # for name, param in self.Qformer.named_parameters():
        #     if "_query" in name:
        #         key_orig = name.replace("_query", "")
        #         param.data.copy_(state_dict[key_orig])

        # import pdb; pdb.set_trace()
        self.clip_model, clip_preprocess = clip.load("RN50", jit=False)

        self.combiner = Combiner(
            clip_feature_dim=self.clip_model.visual.output_dim,
            projection_dim=640 * 4,
            hidden_dim=640 * 8,
        )
        # self.combiner = CrossAttentionCombiner(feature_dim=self.clip_model.visual.output_dim, num_heads=8, dropout= 0.1)

        if si_el_weight > 0:
            logging.info("using elimination")
        if si_att_weight > 0:
            logging.info("using attraction")
        if si_ti_weight > 0:
            logging.info("using original loss only")

        # self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        # self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = temperature

        self.max_txt_len = max_txt_len

        # for p in self.vision_proj.parameters():
        #     p.requires_grad = False

        # for p in self.ln_vision.parameters():
        #     p.requires_grad = False

        # for p in self.Qformer.cls.parameters():
        #     p.requires_grad = False

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # logging.info('Only the CLIP text encoder will be fine-tuned')
        # for param in self.clip_model.visual.parameters():
        #     param.requires_grad = False

        assert (
            si_ti_weight + si_tc_weight + si_el_weight + si_att_weight > 0
        ), "No loss term is enabled"

        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight
        self.si_el_weight = si_el_weight
        self.si_att_weight = si_att_weight

    def forward(self, batch, fabric):
        ref_img = batch["ref_img"]
        tar_img_feat = batch["tar_img_feat"]
        caption = batch["edit"]

        source_labels = batch["source_label"]
        target_labels = batch["target_label"]

        N = len(source_labels)

        false_negatives = build_false_negative_dict(
            zip(source_labels, target_labels), N
        )
        false_negatives_indices = []
        for i in source_labels:
            indices = false_negatives[i]
            false_negatives_indices.append(indices)

        pos_dict = {}
        # import pdb; pdb.set_trace()
        for i, pair in enumerate(false_negatives_indices):
            if len(pair) > 1:
                pos_dict[i] = pair

        # pos_pairs = []
        # for i, (src,trt) in enumerate(zip(source_labels, target_labels)):
        #     pos_pairs.append((i, j))

        ref_img.half()

        device = ref_img.device

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)

        # Text
        # text_tokens = self.tokenizer(
        #     caption,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(device)

        # import pdb; pdb.set_trace()
        caption = clip.tokenize(caption, truncate=True).to(device)
        text_inputs_list = torch.split(caption, 32)
        text_features = torch.vstack(
            [
                self.clip_model.encode_text(mini_batch).float()
                for mini_batch in text_inputs_list
            ]
        )

        query_si_feat = self.combiner(ref_img, text_features)  # normalized features
        query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        tar_img_feat = F.normalize(tar_img_feat, dim=-1)

        # import pdb; pdb.set_trace()
        # if self.train_vit:
        #     ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        # else:
        #     with torch.no_grad():
        #         ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

        # Encode the reference image
        # ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        # ###============== Image-text Matching ===================###
        # query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     self.device
        # )
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        # output = self.Qformer.bert(
        #     text_tokens.input_ids,  # [bs, 32]
        #     query_embeds=query_tokens,  # [bs, 32, 768]
        #     attention_mask=attention_mask,  # [bs, 64]
        #     encoder_hidden_states=ref_img_embs,  # [bs, 677, 1408]
        #     encoder_attention_mask=ref_img_atts,  # [bs, 677]
        #     return_dict=True,
        # )

        # vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        # query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
        # query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        # mean over all query tokens
        # query_si_feat = query_si_feat.mean(dim=1)
        # tar_img_feat = tar_img_feat.mean(dim=1)

        # import pdb; pdb.set_trace()

        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.si_ti_weight > 0:
            si_ti_loss, sim_matrix = self.loss(query_si_feat, tar_img_feat, self.temp)
            # si_ti_loss, sim_matrix = self.loss(query_si_feat, tar_imig_feat, self.temp, false_negatives_indices)
            # si_ti_loss_elim = self.loss_fn_elimination(query_si_feat, tar_img_feat, self.temp, false_negatives_indices, pos_dict=None)
            # si_ti_loss_attract = self.loss_fn_attraction(query_si_feat, tar_img_feat, self.temp, false_negatives_indices, pos_dict=pos_dict)

            # si_ti_loss =  1.0 * si_ti_loss_elim + 0.0 * si_ti_loss_attract
            loss += si_ti_loss * self.si_ti_weight

            # Convert the tensor to a numpy array
            # sim_matrix = sim_matrix.cpu().detach().numpy()

        if self.si_el_weight > 0:
            si_ti_loss_elim = self.loss_fn_elimination(
                query_si_feat,
                tar_img_feat,
                self.temp,
                false_negatives_indices,
                pos_dict=None,
            )

            loss += si_ti_loss_elim * self.si_el_weight

        if self.si_att_weight > 0:
            si_ti_loss_attract = self.loss_fn_attraction(
                query_si_feat,
                tar_img_feat,
                self.temp,
                false_negatives_indices,
                pos_dict=pos_dict,
            )

            loss += si_ti_loss_attract * self.si_att_weight

        if self.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]

            tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)

            si_tc_loss = self.loss(query_si_feat, tar_txt_feat, self.temp)
            loss += si_tc_loss * self.si_tc_weight

        return loss


def blip2_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model
