import contextlib
import datetime
import logging
import os
import time

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger
from lavis.common.utils import is_url
from typing import Any
from src.tools.utils import all_gather_with_grad, concat_all_gather
from src.model.aim.aim_model import ViT_CLIP


class AIM(nn.Module):
    def __init__(
        self,
        # loss: Any,
        n_classes=4,
        train_model=False,
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
        **kwargs,
    ):
        super(AIM, self).__init__()

        self.loss = nn.CrossEntropyLoss()

        self.model = ViT_CLIP(
        input_resolution=224,
        patch_size=16,
        num_frames=12,
        width=768,
        layers=12,
        heads=12,
        drop_path_rate=0.1, 
        # in_feat_size = 768,
        num_classes=306,
        )
        
        # self.fc = nn.Linear(512, 4)

        # if not train_model:
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        #     self.model = self.model.eval()
        #     self.model.train = self.disabled_train
        #     logging.info("freezing model")

    def forward(self, input, fabric):
        videos_batch = input["video"]
        labels = input["label"]
        outputs = self.model(videos_batch)

        outputs = all_gather_with_grad(outputs, fabric)

        loss = 0

        loss += self.loss(
            outputs,
            labels,
        )

        return loss
    
    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self
    
    def load_from_pretrained(self, url_or_filename):
        # import pdb; pdb.set_trace()
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        elif os.path.isfile(os.path.join(get_original_cwd(), url_or_filename)):
            checkpoint = torch.load(
                os.path.join(get_original_cwd(), url_or_filename), map_location="cpu"
            )
        else:
            raise RuntimeError(f"checkpoint url or path is invalid: {url_or_filename}")

        # state_dict = checkpoint["model"]
        # updated_ckpt = {k.replace("backbone", "model"): v for k, v in checkpoint.items()}
        # del updated_ckpt['cls_head.fc_cls.weight'], updated_ckpt["cls_head.fc_cls.bias"]

        # msg = self.load_state_dict(updated_ckpt, strict=False)
        msg = self.load_state_dict(checkpoint, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
    
def aim_pretrained(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model

        