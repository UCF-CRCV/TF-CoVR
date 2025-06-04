import datetime
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F

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
    # # pair_ids = []
    # query_ids = []
    # query_feats, tar_img_feats, ref_img_ids, tar_img_ids, pair_ids = [], [], [], [], []
    # retrieval_results = {}
    query_feats = []
    query_ids = []
    ref_img_ids = []

    # import pdb; pdb.set_trace()
    for batch in data_loader:
        ref_img = batch["reference_img"]
        caption = batch["relative_caption"]
        # tar_feat = batch["tar_img_feat"]
        # caption = batch["edit"]
        # pair_id = batch["pair_id"]

        query_ids.extend(batch["query_id"])
        ref_img_ids.extend(batch["reference_img_id"])
        # pair_ids.extend(pair_id)
        # captions.extend(caption)

        device = ref_img.device

        # ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
        # ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        # # Text
        # text_tokens = model.tokenizer(
        #     caption,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=model.max_txt_len,
        #     return_tensors="pt",
        # ).to(device)

        # ###============== Image-text Matching ===================###
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

        ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
        image_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state
        vl_feat = F.normalize(model.vision_proj(image_embeds), dim=-1)
        # vl_feat = all_gather_with_grad(query_feat, fabric)
        query_feats.append(vl_feat.cpu())

        # Encode the target image
        # tar_img_feats.append(tar_feat.cpu())

    query_feats = torch.cat(query_feats, dim=0)
    ref_img_ids = torch.tensor([int(id) for id in ref_img_ids], dtype=torch.long)
    query_ids = torch.tensor([int(id) for id in query_ids], dtype=torch.long)

    # tar_img_feats = torch.cat(tar_img_feats, dim=0)
    tar_img_feats = data_loader.dataset.embs.to(query_feats.device)
    tar_ids = torch.Tensor(data_loader.dataset.tar_ids).long()

    query_feats = F.normalize(query_feats, dim=-1)
    tar_img_feats = F.normalize(tar_img_feats, dim=-1)

    # ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
    # tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

    # ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
    # tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

    if fabric.world_size > 1:
        # Gather tensors from every process
        query_feats = fabric.all_gather(query_feats)
        # tar_img_feats = fabric.all_gather(tar_img_feats)
        ref_img_ids = fabric.all_gather(ref_img_ids)
        query_ids = fabric.all_gather(query_ids)
        # tar_img_ids = fabric.all_gather(tar_img_ids)

        query_feats = einops.rearrange(query_feats, "d b q e -> (d b) q e")
        # tar_img_feats = einops.rearrange(tar_img_feats, "d b q e -> (d b) q e")
        ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
        query_ids = einops.rearrange(query_ids, "d b -> (d b)")
        # tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

    if fabric.global_rank == 0:
        ref_img_ids = ref_img_ids.cpu().numpy().tolist()
        assert len(ref_img_ids) == len(query_feats)
        assert len(ref_img_ids) == len(query_ids)

        query_feats = query_feats.cpu()
        tar_img_feats = tar_img_feats.cpu()

        # import pdb; pdb.set_trace()
        query_feats = query_feats.mean(dim=1)
        # query_feats = query_feats.mean(dim=1)
        # tar_img_feats = tar_img_feats.mean(dim=1)
        tar_img_feats = tar_img_feats.mean(dim=1)

        sims_q2t = query_feats @ tar_img_feats.T

        # Set the similarity scores to -100 where query_id == tar_id
        ref_ids = torch.Tensor(ref_img_ids).long()
        mask = (ref_ids[:, None] == tar_ids[None, :]).float()
        sims_q2t -= 100 * mask
        # assert (sims_q2t < -10).sum().item() == len(
        #     ref_ids
        # ), "Not all ref_ids are in the target set"
        sims_q2t = sims_q2t.cpu().numpy()
        tar_ids = tar_ids.cpu().numpy()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Evaluation time {total_time_str}")

        recalls = {}
        assert len(sims_q2t) == len(ref_img_ids)
        assert len(sims_q2t) == len(query_ids)
        for query_id, query_sims in zip(query_ids, sims_q2t):
            sorted_indices = np.argsort(query_sims)[::-1]

            query_id_recalls = list(tar_ids[sorted_indices][:50])
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

    # tar_img_feats = tar_img_feats.mean(dim=1)
    # query_feats = query_feats.mean(dim=1)
    # sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

    # # Add zeros where ref_img_id == tar_img_id
    # for i in range(len(ref_img_ids)):
    #     for j in range(len(tar_img_ids)):
    #         if ref_img_ids[i] == tar_img_ids[j]:
    #             sim_q2t[i][j] = -10

    # recalls = {}
    # assert len(sim_q2t) == len(query_ids)
    # for query_id, query_sims in zip(query_ids, sim_q2t):
    #     sorted_indices = np.argsort(query_sims)[::-1]

    #     indices = sorted_indices[:50].tolist()
    #     # query_id_recalls = list(data_loader.dataset.pairid2tar[sorted_indices][:50])
    #     query_id_recalls = [data_loader.dataset.pairid2tar[idx] for idx in indices]
    #     query_id_recalls = [int(id) for id in query_id_recalls]
    #     recalls[str(query_id.item())] = query_id_recalls

    # # import pdb; pdb.set_trace()
    # ap_atk, recall_atk = compute_metrics(data_loader.dataset, recalls)
    # recalls = {}
    # for k, v in ap_atk.items():
    #     recalls[f"mAP@{k}"] = v
    #     print(f"mAP@{k}: {v:.2f}")

    # for k, v in recall_atk.items():
    #     recalls[f"Recall@{k}"] = v
    #     print(f"Recall@{k}: {v:.2f}")

    # json_dump(recalls, "recalls_circo-val.json")

    # print(f"Recalls saved in {Path.cwd()}/recalls_circo-val.json")

    # ranked_indices = np.argsort(sim_q2t, axis=1)[:, ::-1]

    # # import pdb; pdb.set_trace()
    # for i, pair_id in enumerate(pair_ids):
    #     ref_img_path_idx = data_loader.dataset.pairid2ref[pair_id]
    #     ref_img_path = data_loader.dataset.int2id[ref_img_path_idx]
    #     retrieved_filenames = [data_loader.dataset.int2id_tar[data_loader.dataset.pairid2tar[idx]] for idx in ranked_indices[i,:]]
    #     query_filename = ref_img_path
    #     # retrieved_filenames = [Path(img_path).name for img_path in retrieved_img_paths]
    #     ground_truth_filename = data_loader.dataset.int2id_tar[data_loader.dataset.pairid2tar[pair_id]]

    #     found_in_recall = {
    #         "R1": ground_truth_filename in retrieved_filenames[:1],
    #         "R5": ground_truth_filename in retrieved_filenames[:5],
    #         "R10": ground_truth_filename in retrieved_filenames[:10],
    #         "R50": ground_truth_filename in retrieved_filenames[:50]
    #     }

    #     retrieval_results[str(pair_id)] = {
    #         "query_file": query_filename,
    #         "ground_truth": ground_truth_filename,
    #         # "rank": int(np.where(np.array(retrieved_filenames) == ground_truth_filename)[0][0]) if ground_truth_filename in retrieved_filenames else -1,
    #         "R1": retrieved_filenames[:1],
    #         "R5": retrieved_filenames[:5],
    #         "R10": retrieved_filenames[:10],
    #         "R50": retrieved_filenames[:50],
    #         "found_in_recall": found_in_recall,
    #     }

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print("Evaluation time {}".format(total_time_str))

    # eval_result = eval_recall(sim_q2t)
    # fabric.print(eval_result)

    # # fabric.log_dict(
    # #     {
    # #         "val/R1": eval_result["R1"],
    # #         "val/R5": eval_result["R5"],
    # #         "val/R10": eval_result["R10"],
    # #         "val/R_mean": eval_result["R_mean"],
    # #     }
    # # )

    # metrics = {
    # "val/R1": eval_result["R1"],
    # "val/R5": eval_result["R5"],
    # "val/R10": eval_result["R10"],
    # "val/R_mean": eval_result["R_mean"],
    # }

    # fabric.log_dict(metrics, step=epoch)

    # eval_result = {k: round(v, 2) for k, v in eval_result.items()}
    # eval_result["time"] = total_time_str

    # eval_result["annotation"] = Path(data_loader.dataset.annotation_pth).name
    # annotation_name = Path(data_loader.dataset.annotation_pth).stem
    # json_dump(eval_result, f"eval-recalls_{annotation_name}.json")
    # json_dump(retrieval_results, f"val_retrieval_results_{annotation_name}.json")

    # else:

    # Placeholder for ranks other than 0

    #     eval_result = None

    # # Synchronize across all ranks
    # eval_result = fabric.broadcast(eval_result, src=0)

    # fabric.barrier()

    # return eval_result


# @torch.no_grad()
# def compute_metrics(
#     dataset, predictions_dict: Dict[int, List[int]], ranks: List[int] = [5, 10, 25, 50]
# ) -> Tuple[Dict[int, float], Dict[int, float]]:
#     """Computes the Average Precision (AP) and Recall for a given set of predictions.

#     Args:
#         data_path (Path): Path where the CIRCO datasset is located
#         predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
#         ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

#     Returns:
#         Tuple[Dict[int, float], Dict[int, float]]: Dictionaries with the AP and Recall for each rank
#     """

#     # Initialize empty dictionaries to store the AP and Recall values for each rank
#     aps_atk = defaultdict(list)
#     recalls_atk = defaultdict(list)

#     import pdb; pdb.set_trace()
#     # Iterate through each query id and its corresponding predictions
#     for query_id, predictions in predictions_dict.items():
#         # ref_img_path_idx = dataset.pairid2ref[int(query_id)]
#         ref_img_path = dataset.int2id[int(query_id)]
#         retrieved_filenames = [dataset.int2id_tar[dataset.pairid2tar[idx]] for idx in predictions]
#         target = dataset.get_target_video_ids(ref_img_path)
#         gt_img_ids = target["gt_video_ids"]
#         target_img_id = target["target_video_id"]

#         # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

#         predictions = np.array(predictions, dtype=int)
#         ap_labels = np.isin(predictions, gt_img_ids)
#         precisions = (
#             np.cumsum(ap_labels, axis=0) * ap_labels
#         )  # Consider only positions corresponding to GTs
#         precisions = precisions / np.arange(
#             1, ap_labels.shape[0] + 1
#         )  # Compute precision for each position

#         # Compute the AP and Recall for the given ranks
#         for rank in ranks:
#             aps_atk[rank].append(
#                 float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank))
#             )

#         recall_labels = predictions == target_img_id
#         for rank in ranks:
#             recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

#     # Compute the mean AP and Recall for each rank and store them in a dictionary
#     ap_atk = {}
#     recall_atk = {}
#     for rank in ranks:
#         ap_atk[rank] = round(float(np.mean(aps_atk[rank])) * 100, 2)
#         recall_atk[rank] = round(float(np.mean(recalls_atk[rank])) * 100, 2)
#     return ap_atk, recall_atk


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "R1": round(tr1, 4),
        "R5": round(tr5, 4),
        "R10": round(tr10, 4),
        "R50": round(tr50, 4),
        "R_mean": round(tr_mean, 4),
    }
    return eval_result
