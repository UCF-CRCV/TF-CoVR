import datetime
import time
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from src.tools.files import json_dump


class TestWebVidCoVR:
    def __init__(self, remove_self_similarity: bool = True, dataset: str = "covr"):
        self.remove_self_similarity = remove_self_similarity
        self.dataset = dataset

    @torch.no_grad()
    def __call__(self, model, data_loader, fabric):
        model.eval()

        fabric.print("Computing features for evaluation...")
        start_time = time.time()

        # tar_img_feats = []
        # query_feats = []
        captions = []
        # pair_ids = []
        query_feats, tar_img_feats, ref_img_ids, tar_img_ids, pair_ids = (
            [],
            [],
            [],
            [],
            [],
        )
        retrieval_results = {}

        # import pdb; pdb.set_trace()
        for batch in data_loader:
            ref_img = batch["ref_img"]
            tar_feat = batch["tar_img_feat"]
            caption = batch["edit"]
            pair_id = batch["pair_id"]

            pair_ids.extend(pair_id.cpu().numpy().tolist())
            captions.extend(caption)

            device = ref_img.device

            # ref_img_embs = model.ln_vision(model.visual_encoder(ref_img))
            # ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
            #     device
            # )

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

            text_tokens = model.tokenizer(
                caption,
                padding="longest",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)

            # Shift encoder
            # query_tokens = model.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
            # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            #     device
            # )
            # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            # query_embs = model.Qformer.bert(
            #     text_tokens.input_ids,
            #     query_embeds=query_tokens,
            #     attention_mask=attention_mask,
            #     encoder_hidden_states=ref_img_embs,
            #     encoder_attention_mask=ref_img_atts,
            #     return_dict=True,
            # )
            # query_feat = query_embs.last_hidden_state[:, : query_tokens.size(1), :]
            # query_feat = F.normalize(model.text_proj(query_feat), dim=-1)
            # query_feats.append(query_feat.cpu())

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
            vl_feat = F.normalize(model.text_proj(vl_embs), dim=-1)

            # Reshape back to (bs, num_frames, query_tokens, embedding_dim)
            vl_feat = vl_feat.view(bs, num_frames, query_tokens.size(1), -1)
            vl_feat = vl_feat.mean(dim=1)  # Mean across frames

            query_feats.append(vl_feat.cpu())

            # Encode the target image
            tar_img_feats.append(tar_feat.cpu())

        query_feats = torch.cat(query_feats, dim=0)
        tar_img_feats = torch.cat(tar_img_feats, dim=0)

        query_feats = F.normalize(query_feats, dim=-1)
        tar_img_feats = F.normalize(tar_img_feats, dim=-1)

        ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
        tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

        ref_img_ids = torch.tensor(ref_img_ids, dtype=torch.long)
        tar_img_ids = torch.tensor(tar_img_ids, dtype=torch.long)

        if fabric.world_size > 1:
            # Gather tensors from every process
            query_feats = fabric.all_gather(query_feats)
            tar_img_feats = fabric.all_gather(tar_img_feats)
            ref_img_ids = fabric.all_gather(ref_img_ids)
            tar_img_ids = fabric.all_gather(tar_img_ids)

            query_feats = einops.rearrange(query_feats, "d b q e -> (d b) q e")
            tar_img_feats = einops.rearrange(tar_img_feats, "d b q e -> (d b) q e")
            ref_img_ids = einops.rearrange(ref_img_ids, "d b -> (d b)")
            tar_img_ids = einops.rearrange(tar_img_ids, "d b -> (d b)")

        if fabric.global_rank == 0:
            tar_img_feats = tar_img_feats.mean(dim=1)
            query_feats = query_feats.mean(dim=1)
            sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

            if self.remove_self_similarity:
                for i in range(len(ref_img_ids)):
                    for j in range(len(tar_img_ids)):
                        if ref_img_ids[i] == tar_img_ids[j]:
                            sim_q2t[i][j] = -10

            ranked_indices = np.argsort(sim_q2t, axis=1)[:, ::-1]

            # import pdb; pdb.set_trace()
            for i, pair_id in enumerate(pair_ids):
                ref_img_path_idx = data_loader.dataset.pairid2ref[pair_id]
                ref_img_path = data_loader.dataset.int2id[ref_img_path_idx]
                retrieved_filenames = [
                    data_loader.dataset.int2id_tar[data_loader.dataset.pairid2tar[idx]]
                    for idx in ranked_indices[i, :]
                ]
                query_filename = ref_img_path
                # retrieved_filenames = [Path(img_path).name for img_path in retrieved_img_paths]
                ground_truth_filename = data_loader.dataset.int2id_tar[
                    data_loader.dataset.pairid2tar[pair_id]
                ]

                found_in_recall = {
                    "R1": ground_truth_filename in retrieved_filenames[:1],
                    "R5": ground_truth_filename in retrieved_filenames[:5],
                    "R10": ground_truth_filename in retrieved_filenames[:10],
                    "R50": ground_truth_filename in retrieved_filenames[:50],
                }

                retrieval_results[str(pair_id)] = {
                    "query_file": query_filename,
                    "ground_truth": ground_truth_filename,
                    # "rank": int(np.where(np.array(retrieved_filenames) == ground_truth_filename)[0][0]) if ground_truth_filename in retrieved_filenames else -1,
                    "R1": retrieved_filenames[:1],
                    "R5": retrieved_filenames[:5],
                    "R10": retrieved_filenames[:10],
                    "R50": retrieved_filenames[:50],
                    "found_in_recall": found_in_recall,
                }

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Evaluation time {}".format(total_time_str))

            recalls = eval_recall(sim_q2t)
            recalls["annotation"] = Path(data_loader.dataset.annotation_pth).name
            fabric.print(recalls)

            # Save results
            self_sim = "" if self.remove_self_similarity else "_ss"
            json_dump(recalls, f"recalls_{self.dataset}{self_sim}.json")
            json_dump(
                retrieval_results,
                f"test_retrieval_results_{self.dataset}{self_sim}.json",
            )

            print(
                f"Recalls saved in {Path.cwd()}/recalls_{self.dataset}{self_sim}.json"
            )

        fabric.barrier()


@torch.no_grad()
def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        # if len(score) == 0:  # Check if score array is empty
        # print(f"Warning: Empty score array at index {index}")
        # continue

        # inds = np.argsort(score)[::-1]  # Get sorted indices in descending order
        # if index not in inds:  # Ensure index exists in inds
        #     print(f"Warning: index {index} not found in inds")
        #     continue

        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # import pdb; pdb.set_trace()
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result


# @torch.no_grad()
# def eval_recall(scores_q2t):
#     """
#     Evaluates recall at different ranks (R@1, R@5, R@10, R@50) for query-to-target matching.

#     Args:
#         scores_q2t (np.ndarray): A 2D array where each row contains similarity scores for a query.

#     Returns:
#         dict: Recall metrics at various levels.
#     """
#     if not isinstance(scores_q2t, np.ndarray):
#         scores_q2t = np.array(scores_q2t, dtype=object)  # Ensure NumPy array format

#     num_queries = scores_q2t.shape[0]
#     ranks = np.full(num_queries, np.inf)  # Initialize with large values

#     for index, score in enumerate(scores_q2t):
#         if score is None or len(score) == 0 or np.isnan(score).all():
#             print(f"Warning: Empty or invalid score array at index {index}")
#             continue  # Skip invalid queries

#         # Convert to valid numerical scores
#         score = np.nan_to_num(score)  # Replace NaN values with zeros
#         inds = np.argsort(score)[::-1]  # Sort scores in descending order

#         if index not in inds:
#             print(f"Warning: index {index} not found in sorted indices.")
#             continue

#         ranks[index] = np.where(inds == index)[0][0]

#     # Remove skipped queries (ranks with `inf` values)
#     valid_ranks = ranks[ranks != np.inf]

#     if len(valid_ranks) == 0:
#         print("Error: No valid queries were found!")
#         return {
#             "R1": 0.0, "R5": 0.0, "R10": 0.0, "R50": 0.0, "R_mean": 0.0
#         }

#     # Compute recall percentages
#     tr1 = 100.0 * np.mean(valid_ranks < 1)
#     tr5 = 100.0 * np.mean(valid_ranks < 5)
#     tr10 = 100.0 * np.mean(valid_ranks < 10)
#     tr50 = 100.0 * np.mean(valid_ranks < 50)

#     tr_mean = (tr1 + tr5 + tr10) / 3  # Mean of top recalls

#     eval_result = {
#         "R1": round(tr1, 4),
#         "R5": round(tr5, 4),
#         "R10": round(tr10, 4),
#         "R50": round(tr50, 4),
#         "R_mean": round(tr_mean, 4),
#     }

#     return eval_result
