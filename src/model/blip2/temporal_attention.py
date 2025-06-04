import torch
import torch.nn as nn
import math


class TemporalAttentionWithPositionEmbedding(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_frames, num_tokens, output_dim, dropout=0.1
    ):
        super(TemporalAttentionWithPositionEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_tokens = num_tokens

        # Positional Embedding for frames
        self.position_embedding = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # Temporal Attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Linear layer for final output
        self.fc = nn.Linear(num_frames * num_tokens * embed_dim, output_dim)

    def forward(self, x):
        # x shape: (B, F, Tokens, emb)
        B, F, T, E = x.shape

        # Add positional embedding to frames
        x = x + self.position_embedding  # Broadcasting over batch and tokens

        # Reshape to (B * Tokens, F, emb) for temporal attention
        x = x.view(B * T, F, E)

        # Apply temporal attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.layer_norm(x)

        # Reshape back to (B, F, Tokens, emb)
        x = x.view(B, T, F, E)
        x = x.permute(0, 2, 1, 3)  # (B, F, Tokens, emb)

        # Flatten and apply linear layer
        x = x.reshape(B, -1)  # (B, F * Tokens * emb)
        x = self.fc(x)  # (B, output_dim)

        return x


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total_params, total_params_trainable


def format_params(num_params):
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


if __name__ == "__main__":

    # Example usage
    B, F, Tokens, emb = (
        16,
        8,
        10,
        512,
    )  # Batch size, Frames, Tokens, Embedding dimension
    output_dim = 256  # Desired output dimension
    num_heads = 8

    model = TemporalAttentionWithPositionEmbedding(
        embed_dim=emb,
        num_heads=num_heads,
        num_frames=F,
        num_tokens=Tokens,
        output_dim=output_dim,
    )

    total_params, total_params_trainable = count_parameters(model)
    print(f"Total parameters: {format_params(total_params)}")
    print(f"Trainable parameters: {format_params(total_params_trainable)}")

    fused_features = torch.randn(B, F, Tokens, emb)  # Example input
    output = model(fused_features)  # Output shape: (B, output_dim)
    print("Output shape:", output.shape)  # Should be (B, output_dim)
