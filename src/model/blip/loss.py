import torch
import torch.nn as nn
import torch.nn.functional as F


class FalseNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, mode="elimination", **kwargs):
        """
        Args:
            temperature (float): Temperature for scaling logits.
            mode (str): 'elimination' or 'attraction'
        """
        super().__init__()
        assert mode in [
            "elimination",
            "attraction",
        ], "mode must be 'elimination' or 'attraction'"
        self.temperature = temperature
        self.mode = mode
        self.beta = 0.5
        self.alpha = 1.0

    def compute_similarity_matrix(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        return torch.matmul(embeddings, embeddings.T)

    def build_false_negative_dict(self, pairs, N):
        fn_dict = defaultdict(set)
        for i, f in pairs:
            fn_dict[int(i)].add(int(f))
        return [fn_dict[i] for i in range(N)]

    # def forward(self, video_embs, text_embs, temp, false_negative_pairs, pos_dict):
    #     """
    #     Args:
    #         video_embs (Tensor): shape (N, D)
    #         text_embs (Tensor): shape (N, D)
    #         false_negative_pairs: list of sets, false_negative_pairs[i] contains indices of false negatives for i
    #         pos_dict: dict mapping index to its positive match
    #     """
    #     N = video_embs.size(0)
    #     sim_matrix = video_embs @ text_embs.T / self.temperature
    #     loss = 0.0

    #     # --- Row-wise: video as anchor ---
    #     for i in range(N):
    #         if self.mode == 'elimination':
    #             numerator = torch.exp(sim_matrix[i, i])
    #             mask = torch.ones(N, dtype=torch.bool, device=video_embs.device)
    #             mask[i] = False
    #             for fn in false_negative_pairs[i]:
    #                 mask[fn] = False
    #             denominator = torch.exp(sim_matrix[i, mask]).sum()
    #             loss += -torch.log(numerator / (denominator + 1e-8))

    #         elif self.mode == 'attraction':
    #             if i not in pos_dict:
    #                 continue
    #             F_i = false_negative_pairs[i]
    #             mask = torch.ones(N, dtype=torch.bool, device=video_embs.device)
    #             mask[i] = False
    #             denom = torch.exp(sim_matrix[i, mask]).sum()
    #             main_term = torch.log(torch.exp(sim_matrix[i, i]) / (denom + 1e-8))
    #             attraction_term = 0.0
    #             for f in F_i:
    #                 attraction_term += torch.log(torch.exp(sim_matrix[i, f]) / (denom + 1e-8))
    #             loss_i = -(main_term + attraction_term) / (1 + len(F_i))
    #             loss += loss_i

    #     # --- Column-wise: text as anchor ---
    #     for j in range(N):
    #         if self.mode == 'elimination':
    #             numerator = torch.exp(sim_matrix[j, j])
    #             mask = torch.ones(N, dtype=torch.bool, device=video_embs.device)
    #             mask[j] = False
    #             for fn in false_negative_pairs[j]:
    #                 mask[fn] = False
    #             denominator = torch.exp(sim_matrix[mask, j]).sum()
    #             loss += -torch.log(numerator / (denominator + 1e-8))

    #         elif self.mode == 'attraction':
    #             if j not in pos_dict:
    #                 continue
    #             F_j = false_negative_pairs[j]
    #             mask = torch.ones(N, dtype=torch.bool, device=video_embs.device)
    #             mask[j] = False
    #             denom = torch.exp(sim_matrix[mask, j]).sum()
    #             main_term = torch.log(torch.exp(sim_matrix[j, j]) / (denom + 1e-8))
    #             attraction_term = 0.0
    #             for f in F_j:
    #                 attraction_term += torch.log(torch.exp(sim_matrix[f, j]) / (denom + 1e-8))
    #             loss_j = -(main_term + attraction_term) / (1 + len(F_j))
    #             loss += loss_j

    #     return loss / (2 * N)

    def forward(
        self, video_embs, text_embs, temp, false_negative_pairs=None, pos_dict=None
    ):
        N = video_embs.size(0)
        sim_matrix = video_embs @ text_embs.T / self.temperature
        sim_matrix = sim_matrix.float()
        beta_sim = self.beta * sim_matrix

        # --- Compute base weights ---
        exp_beta_sim = torch.exp(beta_sim)

        w_v2t = (
            (N - 1)
            * exp_beta_sim
            / (
                exp_beta_sim.sum(dim=1, keepdim=True)
                - torch.exp(torch.diagonal(beta_sim)).unsqueeze(1)
            )
        )
        w_t2v = (
            (N - 1)
            * exp_beta_sim
            / (
                exp_beta_sim.sum(dim=0, keepdim=True)
                - torch.exp(torch.diagonal(beta_sim)).unsqueeze(0)
            )
        )

        # Set diagonal entries to alpha
        diag_idx = torch.arange(N)
        w_v2t[diag_idx, diag_idx] = self.alpha
        w_t2v[diag_idx, diag_idx] = self.alpha

        loss = 0.0

        # --- Row-wise: video as anchor ---
        for i in range(N):
            mask = torch.ones(N, dtype=torch.bool, device=video_embs.device)
            mask[i] = False

            if self.mode == "elimination":
                for fn in false_negative_pairs[i]:
                    mask[fn] = False
                numerator = torch.exp(sim_matrix[i, i])
                denom = torch.exp(sim_matrix[i]) * w_v2t[i]
                denominator = denom[mask].sum()
                loss += -torch.log(numerator / (denominator + 1e-8))

            elif self.mode == "attraction":
                if i not in pos_dict:
                    continue
                F_i = false_negative_pairs[i]
                for fn in F_i:
                    mask[fn] = False
                denom = torch.exp(sim_matrix[i]) * w_v2t[i]
                denominator = denom[mask].sum()
                main_term = torch.log(
                    torch.exp(sim_matrix[i, i]) / (denominator + 1e-8)
                )
                attraction_term = 0.0
                for f in F_i:
                    attraction_term += torch.log(
                        torch.exp(sim_matrix[i, f]) / (denominator + 1e-8)
                    )
                loss_i = -(main_term + attraction_term) / (1 + len(F_i))
                loss += loss_i

        # --- Column-wise: text as anchor ---
        for j in range(N):
            mask = torch.ones(N, dtype=torch.bool, device=video_embs.device)
            mask[j] = False

            if self.mode == "elimination":
                for fn in false_negative_pairs[j]:
                    mask[fn] = False
                numerator = torch.exp(sim_matrix[j, j])
                denom = torch.exp(sim_matrix[:, j]) * w_t2v[:, j]
                denominator = denom[mask].sum()
                loss += -torch.log(numerator / (denominator + 1e-8))

            elif self.mode == "attraction":
                if j not in pos_dict:
                    continue
                F_j = false_negative_pairs[j]
                for fn in F_j:
                    mask[fn] = False
                denom = torch.exp(sim_matrix[:, j]) * w_t2v[:, j]
                denominator = denom[mask].sum()
                main_term = torch.log(
                    torch.exp(sim_matrix[j, j]) / (denominator + 1e-8)
                )
                attraction_term = 0.0
                for f in F_j:
                    attraction_term += torch.log(
                        torch.exp(sim_matrix[f, j]) / (denominator + 1e-8)
                    )
                loss_j = -(main_term + attraction_term) / (1 + len(F_j))
                loss += loss_j

        return loss / (2 * N)


class CrossEntropyLoss(nn.Module):
    """
    Hard Negative NCE loss for contrastive learning.
    """

    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, tar_img_feat: torch.Tensor, query_feat: torch.Tensor, temp):
        device = tar_img_feat.device

        sim_t2q = tar_img_feat @ query_feat.T / temp
        sim_q2t = query_feat @ tar_img_feat.T / temp

        bs = sim_t2q.size(0)
        loss_t2q = F.cross_entropy(sim_t2q, torch.arange(bs, device=device))
        loss_q2t = F.cross_entropy(sim_q2t, torch.arange(bs, device=device))

        return (loss_t2q + loss_q2t) / 2


class HardNegativeNCE(nn.Module):
    """
    Hard-Negative NCE loss for contrastive learning.
    https://arxiv.org/pdf/2301.02280.pdf
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        temp,
    ):
        """
        Args:
            video_embds: (batch_size, video_embd_dim)
            text_embds: (batch_size, text_embd_dim)
        """
        batch_size = video_embds.size(0)
        # computation of the similarity matrix
        sim_matrix = video_embds @ text_embds.T  # (batch_size, batch_size)
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / temp
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss, sim_matrix
