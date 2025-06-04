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

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather
from src.model.blip2.vtn_helper import VTNLongformerModel, pad_to_window_size_local
from src.model.blip2.temporal_attention import TemporalAttentionWithPositionEmbedding
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from collections import defaultdict
from src.model.blip.loss import FalseNegativeContrastiveLoss


def build_false_negative_dict(pairs, N):
    """
    Converts a list of (i, f) pairs into a dict: i -> set of false negative indices
    """

    fn_dict = defaultdict(set)
    idx_dict = defaultdict(set)
    for idx, (i, f) in enumerate(pairs):
        fn_dict[i].add(idx)

    # pair_idxs = []
    # for idx, (i, f) in enumerate(pairs):
    #     print(i,f)
    #     import pdb; pdb.set_trace()
    #     indices = fn_dict[i]
    #     pair_idxs.extend(indices)

    # import pdb; pdb.set_trace()
    # idx_dict = {idx: v for idx, v in enumerate(fn_dict.values())}

    # for idx, v in enumerate(fn_dict.values()):
    #     idx_dict[idx].add(v)

    # Ensure all indices exist, even if empty
    # return [idx_dict[i] for i in range(N)]
    return fn_dict


class BLIP2Cir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
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
        frames=8,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,
    ):
        super().__init__()

        # self.loss = loss
        self.loss_fn_elimination = FalseNegativeContrastiveLoss(mode="elimination")
        self.loss_fn_attraction = FalseNegativeContrastiveLoss(mode="attraction")

        logging.info("Using elimination and attraction")

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        # import pdb; pdb.set_trace()
        # self.temporal_encoder = TemporalAttentionWithPositionEmbedding(embed_dim=embed_dim, num_heads=8,
        # num_frames=frames, num_tokens=num_query_token,
        # output_dim=embed_dim)

        self.temp = temperature

        self.max_txt_len = max_txt_len

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        for p in self.ln_vision.parameters():
            p.requires_grad = False

        for p in self.Qformer.cls.parameters():
            p.requires_grad = False

        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

    def forward(self, batch, fabric):
        # import pdb; pdb.set_trace()
        ref_img = batch["ref_img"]
        tar_img_feat = batch["tar_img_feat"]
        caption = batch["edit"]

        # target_labels = batch["target_label"].cpu().numpy()

        # # check if all target labels are the unique
        # if len(set(target_labels)) != len(target_labels):
        #     raise ValueError("All target labels must be the unique")

        source_labels = batch["source_label"]
        target_labels = batch["target_label"]

        # Build a vocab in the current batch
        # all_labels = batch["source_label"] + batch["target_label"]
        # unique_labels = sorted(set(all_labels))
        # label_vocab = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        # inv_label_vocab = {idx: lbl for lbl, idx in label_vocab.items()}

        # # Convert to tensor for gathering
        # source_label_tensor = torch.tensor([label_vocab[lbl] for lbl in batch["source_label"]], device=ref_img.device)
        # target_label_tensor = torch.tensor([label_vocab[lbl] for lbl in batch["target_label"]], device=ref_img.device)

        # source_labels = concat_all_gather(source_label_tensor, fabric)
        # target_labels = concat_all_gather(target_label_tensor, fabric)

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

        ref_img.half()

        device = ref_img.device

        # import pdb; pdb.set_trace()
        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)

        # import pdb; pdb.set_trace()
        # temporal encoder on target video under no grad setting
        # with torch.no_grad():
        #     tar_img_feat = self.temporal_encoder(tar_img_feat)

        tar_img_feat = concat_all_gather(tar_img_feat, fabric)

        # Text
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # import pdb; pdb.set_trace()
        if self.train_vit:
            # ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

            ref_img_embs = [
                self.ln_vision(self.visual_encoder(ref_img[:, i, :, :, :]))
                for i in range(ref_img.shape[1])
            ]
            ref_img_embs = torch.stack(ref_img_embs)
            ref_img_embs = ref_img_embs.tranpose(1, 0)
        else:
            with torch.no_grad():

                # with torch.no_grad():
                #     import pdb; pdb.set_trace()
                # Process all frames in parallel using a batch-first approach
                bs, num_frames, c, h, w = ref_img.shape  # Unpacking for clarity
                ref_img_embs = self.visual_encoder(
                    ref_img.view(-1, c, h, w)
                )  # Merge frames into batch
                ref_img_embs = self.ln_vision(ref_img_embs).view(
                    bs, num_frames, *ref_img_embs.shape[1:]
                )  # Reshape back

        # Create attention masks
        ref_img_atts = torch.ones(
            (bs, num_frames, ref_img_embs.size(2)), dtype=torch.long, device=device
        )

        # Expand query tokens across frames
        query_tokens = self.query_tokens.expand(bs, -1, -1)  # Shape: [bs, 32, 768]
        query_atts = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=self.device
        )

        # Concatenate attention masks
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        # Flatten image embeddings and attention masks for batch processing
        ref_img_embs_flat = ref_img_embs.view(
            -1, *ref_img_embs.shape[2:]
        )  # Shape: [bs * num_frames, 677, 1408]
        ref_img_atts_flat = ref_img_atts.view(
            -1, ref_img_embs.size(2)
        )  # Shape: [bs * num_frames, 677]

        # Repeat text inputs for each frame
        text_input_ids = text_tokens.input_ids.repeat_interleave(num_frames, dim=0)
        query_embeds = query_tokens.repeat_interleave(num_frames, dim=0)
        attention_mask_exp = attention_mask.repeat_interleave(num_frames, dim=0)

        # Forward pass through QFormer for all frames at once
        output = self.Qformer.bert(
            text_input_ids,  # Shape: [bs * num_frames, 32]
            query_embeds=query_embeds,  # Shape: [bs * num_frames, 32, 768]
            attention_mask=attention_mask_exp,  # Shape: [bs * num_frames, 64]
            encoder_hidden_states=ref_img_embs_flat,  # Shape: [bs * num_frames, 677, 1408]
            encoder_attention_mask=ref_img_atts_flat,  # Shape: [bs * num_frames, 677]
            return_dict=True,
        )

        # Extract and normalize embeddings
        vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        # query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
        query_si_feat = self.text_proj(vl_embs)

        # Reshape back to (bs, num_frames, query_tokens, embedding_dim)
        query_si_feat = query_si_feat.view(bs, num_frames, query_tokens.size(1), -1)
        # query_si_feat = self.temporal_encoder(query_si_feat)

        # import pdb; pdb.set_trace()
        # Gather once after computing features
        query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        # Compute means
        # import pdb; pdb.set_trace()
        # query_si_feat = query_si_feat.mean(dim=1)  # Mean across frames
        # query_si_feat = query_si_feat.mean(dim=1)  # Mean over query tokens

        # ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        #         ref_img_embs = [self.ln_vision(self.visual_encoder(ref_img[:,i,:,:,:])) for i in range(ref_img.shape[1])]
        #         ref_img_embs = torch.stack(ref_img_embs)
        #         # ref_img_embs = ref_img_embs.transpose(1,0)

        # # Encode the reference image
        # # ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)
        # ref_img_atts = torch.ones(ref_img_embs.size()[1:-1], dtype=torch.long).to(device)

        # ###============== Image-text Matching ===================###
        # # query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1) # [bs, 677, 1408]
        # query_tokens = self.query_tokens.expand(ref_img_embs.shape[1], -1, -1) # [bs, 677, 1408]
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     self.device
        # )
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        # # output = self.Qformer.bert(
        # #     text_tokens.input_ids,  # [bs, 32]
        # #     query_embeds=query_tokens,  # [bs, 32, 768]
        # #     attention_mask=attention_mask,  # [bs, 64]
        # #     encoder_hidden_states=ref_img_embs,  # [bs, 677, 1408]
        # #     encoder_attention_mask=ref_img_atts,  # [bs, 677]
        # #     return_dict=True,
        # # )

        # # import pdb; pdb.set_trace()
        # query_si_feat = []
        # for i in range(ref_img_embs.shape[0]):
        #     output_single = self.Qformer.bert(
        #     text_tokens.input_ids,  # [bs, 32]
        #     query_embeds=query_tokens,  # [bs, 32, 768]
        #     attention_mask=attention_mask,  # [bs, 64]
        #     encoder_hidden_states=ref_img_embs[i,...],  # [bs, 677, 1408]
        #     encoder_attention_mask=ref_img_atts,  # [bs, 677]
        #     return_dict=True,
        #     )

        #     vl_embs = output_single.last_hidden_state[:, : query_tokens.size(1), :]
        #     query_si_feat_single = F.normalize(self.text_proj(vl_embs), dim=-1)
        #     query_si_feat_single = all_gather_with_grad(query_si_feat_single, fabric)

        #     query_si_feat.append(query_si_feat_single)

        # # import pdb; pdb.set_trace()
        # query_si_feat = torch.stack(query_si_feat)

        # # vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        # # query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
        # # query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        # # mean over all query tokens
        # import pdb; pdb.set_trace()
        query_si_feat = query_si_feat.mean(dim=1)  # mean across all frames
        query_si_feat = query_si_feat.mean(dim=1)  # mean over all query tokens
        # tar_img_feat = tar_img_feat.mean(dim=1)

        query_si_feat = F.normalize(query_si_feat, dim=-1)
        tar_img_feat = F.normalize(tar_img_feat, dim=-1)

        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.si_ti_weight > 0:
            # si_ti_loss, sim_matrix = self.loss(query_si_feat, tar_img_feat, self.temp)
            # loss += si_ti_loss * self.si_ti_weight

            # Convert the tensor to a numpy array
            # cos_sim_matrix_np = sim_matrix.cpu().detach().numpy()

            si_ti_loss_elim = self.loss_fn_elimination(
                query_si_feat,
                tar_img_feat,
                self.temp,
                false_negatives_indices,
                pos_dict=None,
            )
            si_ti_loss_attract = self.loss_fn_attraction(
                query_si_feat,
                tar_img_feat,
                self.temp,
                false_negatives_indices,
                pos_dict=pos_dict,
            )

            si_ti_loss = si_ti_loss_elim + si_ti_loss_attract

            loss += si_ti_loss * self.si_ti_weight

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
