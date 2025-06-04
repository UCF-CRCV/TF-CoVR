import torch
from torch import nn
import torch.nn.functional as F


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.logit_scale = 100

    def forward(
        self, image_features: torch.tensor, text_features: torch.tensor
    ) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        # target_features = F.normalize(target_features, dim=-1)

        # logits = self.logit_scale * predicted_features @ target_features.T
        # return logits
        return predicted_features

    def combine_features(
        self, image_features: torch.tensor, text_features: torch.tensor
    ) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(
            F.relu(self.text_projection_layer(text_features))
        )
        image_projected_features = self.dropout2(
            F.relu(self.image_projection_layer(image_features))
        )

        raw_combined_features = torch.cat(
            (text_projected_features, image_projected_features), -1
        )
        combined_features = self.dropout3(
            F.relu(self.combiner_layer(raw_combined_features))
        )
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * image_features
        )
        return F.normalize(output, dim=-1)


class CrossAttentionCombiner(nn.Module):
    """
    Combiner module that fuses textual and visual information using cross-attention.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        :param feature_dim: Dimension of the input features from CLIP (text & image)
        :param num_heads: Number of heads for multi-head attention
        :param dropout: Dropout rate
        """
        super(CrossAttentionCombiner, self).__init__()

        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.text_proj = nn.Linear(feature_dim, feature_dim)
        self.image_proj = nn.Linear(feature_dim, feature_dim)

        self.output_layer = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        :param image_features: (B, D) image features
        :param text_features: (B, D) text features
        :return: (B, D) fused features
        """
        B, D = image_features.size()

        # Add sequence dimension: (B, 1, D)
        image_feat_seq = image_features.unsqueeze(1)
        text_feat_seq = text_features.unsqueeze(1)

        # Text queries image (T→I)
        text2image_output, _ = self.text_to_image_attn(
            query=text_feat_seq, key=image_feat_seq, value=image_feat_seq
        )

        # Image queries text (I→T)
        image2text_output, _ = self.image_to_text_attn(
            query=image_feat_seq, key=text_feat_seq, value=text_feat_seq
        )

        # Remove sequence dimension
        text2image_output = text2image_output.squeeze(1)
        image2text_output = image2text_output.squeeze(1)

        # Combine the two cross-attended outputs
        fused = 0.5 * (text2image_output + image2text_output)

        # Optionally: add residual info (projected) and normalize
        fused = self.output_layer(
            self.dropout(
                self.norm(
                    fused
                    + self.text_proj(text_features)
                    + self.image_proj(image_features)
                )
            )
        )

        # Normalize final output (important for contrastive training)
        return F.normalize(fused, dim=-1)
