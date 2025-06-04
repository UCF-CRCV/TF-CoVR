import math
import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import (
    vit_base_patch16_224,
    vit_base_patch16_224_in21k,
)

# import vtn_helper
from src.model.blip2_frames.vtn_helper import (
    VTNLongformerModel,
    pad_to_window_size_local,
)
import yaml

# from mlp import mlp


def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed contents of the YAML file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


class VTN(nn.Module):
    """
    VTN model builder. It uses ViT-Base as the backbone.
    Daniel Neimark, Omri Bar, Maya Zohar and Dotan Asselmann.
    "Video Transformer Network."
    https://arxiv.org/abs/2102.00719
    """

    def __init__(self, cfg, weight, backbone, pretrained):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VTN, self).__init__()
        self._construct_network(cfg, weight, backbone, pretrained)

    def _construct_network(self, cfg, weight, backbone, pretrained):
        """
        Builds a VTN model, with a given backbone architecture.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        # print("cfg["MODEL"]["ARCH"]", cfg["MODEL"]["ARCH"])
        if cfg["MODEL"]["ARCH"] == "VIT":
            self.backbone = vit_base_patch16_224(
                pretrained=cfg["VTN"]["PRETRAINED"],
                num_classes=0,
                drop_path_rate=cfg["VTN"]["DROP_PATH_RATE"],
                drop_rate=cfg["VTN"]["DROP_RATE"],
            )
        elif cfg["MODEL"]["ARCH"] == "VIT21k":
            self.backbone = vit_base_patch16_224_in21k(
                pretrained=cfg["VTN"]["PRETRAINED"],
                num_classes=0,
                drop_path_rate=cfg["VTN"]["DROP_PATH_RATE"],
                drop_rate=cfg["VTN"]["DROP_RATE"],
            )
        elif cfg["MODEL"]["ARCH"] == "R50":
            # print("cfg["VTN"]["PRETRAINED"]", cfg["VTN"]["PRETRAINED"])
            print("---backbone---", backbone)
            if backbone == "r50":
                self.backbone = torchvision.models.resnet50(pretrained=pretrained)
                if weight != "":
                    pretrained_kvpair = torch.load(weight)["state_dict"]
                    model_kvpair = self.backbone.state_dict()

                    for layer_name, weights in pretrained_kvpair.items():
                        if layer_name[:2] == "0.":
                            layer_name = layer_name[2:]
                        if layer_name[:2] == "1.":
                            # print(f'excluding {layer_name}')
                            continue
                        model_kvpair[layer_name] = weights
                    self.backbone.load_state_dict(model_kvpair, strict=True)
            if backbone == "r18":
                self.backbone = torchvision.models.resnet18(pretrained=pretrained)
            if backbone == "r34":
                self.backbone = torchvision.models.resnet34(pretrained=pretrained)
            self.backbone.fc = nn.Identity()

        # VTN_VIT_B_KINETICS.pyth

        else:
            raise NotImplementedError(f"not supporting {cfg["MODEL"]["ARCH"]}")
        if cfg["MODEL"]["ARCH"] == "VIT":
            embed_dim = 768
        else:
            embed_dim = 2048
        if backbone == "r18":
            embed_dim = 512
        if backbone == "r34":
            embed_dim = 512

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.temporal_encoder = VTNLongformerModel(
            embed_dim=embed_dim,
            max_position_embeddings=cfg["VTN"]["MAX_POSITION_EMBEDDINGS"],
            num_attention_heads=cfg["VTN"]["NUM_ATTENTION_HEADS"],
            num_hidden_layers=cfg["VTN"]["NUM_HIDDEN_LAYERS"],
            attention_mode=cfg["VTN"]["ATTENTION_MODE"],
            pad_token_id=cfg["VTN"]["PAD_TOKEN_ID"],
            attention_window=cfg["VTN"]["ATTENTION_WINDOW"],
            intermediate_size=cfg["VTN"]["INTERMEDIATE_SIZE"],
            attention_probs_dropout_prob=cfg["VTN"]["ATTENTION_PROBS_DROPOUT_PROB"],
            hidden_dropout_prob=cfg["VTN"]["HIDDEN_DROPOUT_PROB"],
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, cfg["VTN"]["MLP_DIM"]),
            nn.GELU(),
            nn.Dropout(cfg["MODEL"]["DROPOUT_RATE"]),
            nn.Linear(cfg["VTN"]["MLP_DIM"], cfg["MODEL"]["NUM_CLASSES"]),
        )

        # self.twoDmlp = mlp(feature_size= embed_dim)
        self.position_id = 0

    def forward(self, x, bboxes=None):

        x, position_ids = x

        # spatial backbone
        B, C, F, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * F, C, H, W)
        x = self.backbone(x)
        # twoDrep = self.twoDmlp(x)

        # max pool over feature --> 5,5,1 --> upscale
        # print(f'Shape after backbone {x.shape}')
        x = x.reshape(B, F, -1)

        # temporal encoder (Longformer)
        B, D, E = x.shape
        attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        attention_mask[:, 0] = 2
        x, attention_mask, position_ids = vtn_helper.pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id,
        )
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=x.device)
        token_type_ids[:, 0] = 1

        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        position_ids[:, 0] = max_position_embeddings - 2
        # print("position_ids")
        # print(position_ids.shape)
        # print("mask")
        # print(mask.shape)
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(
            input_ids=None,
            attention_mask=attention_mask,  # [B, 36]
            token_type_ids=token_type_ids,  # [B, 36]
            position_ids=position_ids,  # [B, 36]
            inputs_embeds=x,  # [B, 36, 768]
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        # MLP head
        x = x["last_hidden_state"]
        x = self.mlp_head(x[:, 0])
        return x, twoDrep


if __name__ == "__main__":
    import numpy as np

    # from torchsummary import summary

    bs = 2
    num_frames = 8
    pos_array = (
        torch.from_numpy(np.asarray(list(range(num_frames))))
        .unsqueeze(0)
        .repeat(bs, 1)
        .cuda()
    )
    # print(pos_array.shape)
    rand_input = [torch.randn(bs, 3, num_frames, 224, 224).cuda(), pos_array]
    # print(rand_input.shape)

    import argparse
    import sys, os

    # from slowfast_config_defaults import get_cfg
    # from parser_sf import parse_args, load_config

    # args = parse_args()
    cfg_file = "eventVIT_B_VTN.yaml"
    # args.cfg_file = 'eventR50_VTN.yaml'
    #
    # cfg = load_config(args)
    cfg = load_yaml(cfg_file)
    # print(cfg) # config read operation seems working for now, not sure what to ignore in the read config

    # print(vtn_model)
    # import importlib
    # spec = importlib.util.spec_from_file_location('params', cfg_file)
    # params = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(params)

    print("main vtn.py calls to print model")

    # vtn_model.load_state_dict(pretrained, strict=True)
    # exit()
    # for layer_name, weights in pretrained_kvpair.items():
    #     if 'temporal_encoder.embeddings' in layer_name:
    #         print(layer_name)
    #     if 'temporal_encoder.embeddings.position_ids' in layer_name:
    #         print(weights)

    # for layer_name, weights in model_kvpair.items():
    #     if 'temporal_encoder.embeddings' in layer_name:
    #         print(layer_name)

    # exit()
    vtn_model = VTN(cfg, "", "", True).cuda()
    # pretrained_kvpair = torch.load('VTN_VIT_B_KINETICS.pyth')['model_state']
    # model_kvpair = vtn_model.state_dict()
    # for layer_name, weights in pretrained_kvpair.items():
    #     # layer_name.replace('position_id','position_embeddings')

    #     if 'mlp_head.4' in layer_name or 'temporal_encoder.embeddings.position_ids' in layer_name or 'temporal_encoder.embeddings.position_embeddings' in layer_name:
    #         print(f'Skipping {layer_name}')
    #         continue
    #     model_kvpair[layer_name] = weights
    #     vtn_model.load_state_dict(model_kvpair, strict=True)
    # print('model loaded successfully')

    # model, [(1, 300), (1, 300)], dtypes=[torch.float, torch.long]

    # print(vtn_model)

    # for name, param in vtn_model.named_parameters():
    #     if 'backbone' in name:
    #         param.requires_grad = False

    # for name, param in vtn_model.named_parameters():
    #     print(name, param.requires_grad)

    # for m in list(model.parameters())[:-2]:
    #     m.requires_grad = False

    # import pdb; pdb.set_trace()
    output = vtn_model(rand_input)
    # print(output[0].shape, output[1].shape)
    print(output[0].shape)

    # print(torch.cuda.memory_allocated()/1e9) #3.433
    # summary(vtn_model,  [(3,16,224,224), (1, 16)], dtypes=[torch.float, torch.float])
    # summary(vtn_model, rand_input)
    # print(output.shape)
