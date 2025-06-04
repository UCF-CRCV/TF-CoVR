import datetime
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from src.model.blip2.vtn_helper import VTNLongformerModel, pad_to_window_size_local


from src.tools.files import json_dump


class TestEvaluate:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def __call__(model, data_loader, fabric):
        evaluate(model, data_loader, fabric)


@torch.no_grad()
def evaluate(model, data_loader, epoch, fabric):
    model.eval()

    fabric.print("Computing features for evaluation...")
    start_time = time.time()

    # tar_img_feats = []
    # query_feats = []
    # captions = []
    # pair_ids = []
    query_feats = []
    query_ids = []
    ref_img_ids = []

    for batch in data_loader:
        ref_img = batch["reference_img"]
        caption = batch["relative_caption"]
        # tar_feat = batch["tar_img_feat"]
        # caption = batch["edit"]
        # pair_id = batch["pair_id"]

        query_ids.extend(batch["query_id"])
        ref_img_ids.extend(batch["reference_img_id"])

        # pair_ids.extend(pair_id.cpu().numpy().tolist())
        # captions.extend(caption)

        device = ref_img.device

        # ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
        # ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        bs, num_frames, c, h, w = ref_img.shape  # Unpacking for clarity
        ref_img_embs = model.visual_encoder(
            ref_img.view(-1, c, h, w)
        )  # Merge frames into batch
        ref_img_embs = model.ln_vision(ref_img_embs).view(
            bs, num_frames, *ref_img_embs.shape[1:]
        )  # Reshape back

        # Create attention masks
        ref_img_atts = torch.ones(
            (bs, num_frames, ref_img_embs.size(2)), dtype=torch.long, device=device
        )

        # Text
        text_tokens = model.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)

        ###============== Image-text Matching ===================###
        # query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # # attention_mask = text_tokens.attention_mask

        # output = model.Qformer.bert(
        #     text_tokens.input_ids,
        #     query_embeds=query_tokens,
        #     attention_mask=attention_mask,
        #     encoder_hidden_states=ref_img_embs,
        #     encoder_attention_mask=ref_img_atts,
        #     return_dict=True,
        # )
        # vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        # vl_feat = F.normalize(model.text_proj(vl_embs), dim=-1)
        # query_feats.append(vl_feat.cpu())

        # Expand query tokens across frames
        query_tokens = model.query_tokens.expand(bs, -1, -1)  # Shape: [bs, 32, 768]
        query_atts = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=device
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
        output = model.Qformer.bert(
            text_input_ids,  # Shape: [bs * num_frames, 32]
            query_embeds=query_embeds,  # Shape: [bs * num_frames, 32, 768]
            attention_mask=attention_mask_exp,  # Shape: [bs * num_frames, 64]
            encoder_hidden_states=ref_img_embs_flat,  # Shape: [bs * num_frames, 677, 1408]
            encoder_attention_mask=ref_img_atts_flat,  # Shape: [bs * num_frames, 677]
            return_dict=True,
        )

        # Extract and normalize embeddings
        vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        # vl_feat = F.normalize(model.text_proj(vl_embs), dim=-1)
        vl_feat = model.text_proj(vl_embs)

        # Reshape back to (bs, num_frames, query_tokens, embedding_dim)
        vl_feat = vl_feat.view(bs, num_frames, query_tokens.size(1), -1)

        # vl_feat = model.temporal_encoder(vl_feat)  # Apply temporal encoder

        # vl_feat = vl_feat.mean(dim=1)  # Mean across frames

        # import pdb; pdb.set_trace()
        # query_si_feat = query_si_feat.mean(dim=1)  # Mean over query tokens

        # import pdb; pdb.set_trace()
        query_feats.append(vl_feat.cpu())

        # Encode the target image
        # tar_feat = model.temporal_encoder(tar_feat)  # Apply temporal encoder
        # tar_img_feats.append(tar_feat.cpu())

    query_feats = torch.cat(query_feats, dim=0)
    # tar_img_feats = torch.cat(tar_img_feats, dim=0)

    # ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
    # tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

    # ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
    # tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

    ref_img_ids = torch.tensor([int(id) for id in ref_img_ids], dtype=torch.long)
    query_ids = torch.tensor([int(id) for id in query_ids], dtype=torch.long)

    # tar_img_feats = torch.cat(tar_img_feats, dim=0)
    tar_img_feats = data_loader.dataset.embs.to(query_feats.device)
    tar_ids = torch.Tensor(data_loader.dataset.tar_ids).long()

    query_feats = F.normalize(query_feats, dim=-1)
    tar_img_feats = F.normalize(tar_img_feats, dim=-1)

    if fabric.world_size > 1:
        # Gather tensors from every process
        query_feats = fabric.all_gather(query_feats)
        tar_img_feats = fabric.all_gather(tar_img_feats)
        ref_img_ids = fabric.all_gather(ref_img_ids)
        tar_img_ids = fabric.all_gather(tar_img_ids)

        query_feats = einops.rearrange(query_feats, "d b q -> (d b) q")
        tar_img_feats = einops.rearrange(tar_img_feats, "d b q -> (d b) q")
        ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
        tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

    if fabric.global_rank == 0:
        # import pdb; pdb.set_trace()
        query_feats = query_feats.mean(dim=1)
        # query_feats = query_feats.mean(dim=1)
        # tar_img_feats = tar_img_feats.mean(dim=1)
        # tar_img_feats = tar_img_feats.mean(dim=1)

        sims_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

        # Add zeros where ref_img_id == tar_img_id
        # for i in range(len(ref_img_ids)):
        #     for j in range(len(tar_img_ids)):
        #         if ref_img_ids[i] == tar_img_ids[j]:
        #             sim_q2t[i][j] = -10

        ref_ids = torch.Tensor(ref_img_ids).long()
        mask = (ref_ids[:, None] == tar_ids[None, :]).float().numpy()
        sims_q2t -= 100 * mask

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Evaluation time {}".format(total_time_str))

        # import pdb; pdb.set_trace()
        recalls = {}
        assert len(sims_q2t) == len(ref_img_ids)
        assert len(sims_q2t) == len(query_ids)
        # import pdb; pdb.set_trace()
        for query_id, query_sims in zip(query_ids, sims_q2t):
            sorted_indices = np.argsort(query_sims)[::-1].copy()

            query_id_recalls = list(tar_ids[sorted_indices][:50].cpu().numpy())
            query_id_recalls = [int(id) for id in query_id_recalls]
            recalls[str(query_id.item())] = query_id_recalls

        ap_atk, recall_atk = compute_metrics(data_loader.dataset, recalls)
        # import pdb; pdb.set_trace()
        recalls = {}
        for k, v in ap_atk.items():
            recalls[f"mAP@{k}"] = v
            print(f"mAP@{k}: {v:.2f}")

        for k, v in recall_atk.items():
            recalls[f"Recall@{k}"] = v
            print(f"Recall@{k}: {v:.2f}")

        fabric.log_dict(recalls, step=epoch)

        json_dump(recalls, "recalls_circo-val.json")

        print(f"Recalls saved in {Path.cwd()}/recalls_circo-val.json")

    else:
        # Placeholder for ranks other than 0
        recalls = None

    # Synchronize across all ranks
    recalls = fabric.broadcast(recalls, src=0)

    fabric.barrier()

    return recalls


@torch.no_grad()
def compute_metrics(
    dataset, predictions_dict: Dict[int, List[int]], ranks: List[int] = [5, 10, 25, 50]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO datasset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: Dictionaries with the AP and Recall for each rank
    """

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    # import pdb; pdb.set_trace()
    for query_id, predictions in predictions_dict.items():
        # print("query_id:", query_id)
        target = dataset.get_target_img_ids(int(query_id))
        gt_img_ids = target["gt_img_ids"]
        target_img_id = target["target_img_id"]

        # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

        predictions = np.array(predictions, dtype=int)
        gt_img_ids = np.array(gt_img_ids, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = (
            np.cumsum(ap_labels, axis=0) * ap_labels
        )  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(
            1, ap_labels.shape[0] + 1
        )  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(
                float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank))
            )

        # import pdb; pdb.set_trace()
        recall_labels = predictions == target_img_id
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    ap_atk = {}
    recall_atk = {}
    for rank in ranks:
        ap_atk[rank] = round(float(np.mean(aps_atk[rank])) * 100, 2)
        recall_atk[rank] = round(float(np.mean(recalls_atk[rank])) * 100, 2)
    return ap_atk, recall_atk
