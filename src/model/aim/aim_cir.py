import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather
from collections import defaultdict
from src.model.clip.combiner import Combiner


class AIMCir(Blip2Base):

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
    ):
        super().__init__()

        self.loss = loss

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 768, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        logging.info("using MLP with AIM")
        self.combiner = Combiner(
            clip_feature_dim=768, projection_dim=768 * 4, hidden_dim=768 * 8
        )

        if si_ti_weight > 0:
            logging.info("using original loss only")

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = temperature

        self.max_txt_len = max_txt_len

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        for p in self.Qformer.cls.parameters():
            p.requires_grad = False

        self.query_tokens.requires_grad = False

        for name, param in self.Qformer.bert.encoder.named_parameters():
            if "crossattention" in name:
                param.requires_grad = False
            if "output_query.dense" in name:
                param.requires_grad = False
            if "output_query.LayerNorm" in name:
                param.requires_grad = False
            if "intermediate_query.dense" in name:
                param.requires_grad = False

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

        ref_img.half()

        device = ref_img.device

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)

        # Text
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        query_embs = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        vl_embs = query_embs.last_hidden_state[:, 0, :]
        vl_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
        vl_feat = all_gather_with_grad(vl_feat, fabric)

        query_si_feat = self.combiner(ref_img, vl_feat)

        query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        query_si_feat = F.normalize(query_si_feat, dim=-1)
        tar_img_feat = F.normalize(tar_img_feat, dim=-1)

        loss = 0
        if self.si_ti_weight > 0:
            si_ti_loss, sim_matrix = self.loss(query_si_feat, tar_img_feat, self.temp)

            loss += si_ti_loss * self.si_ti_weight

        if self.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]

            tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)

            si_tc_loss = self.loss(query_si_feat, tar_txt_feat, self.temp)
            loss += si_tc_loss * self.si_tc_weight

        return loss


def aim_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model
