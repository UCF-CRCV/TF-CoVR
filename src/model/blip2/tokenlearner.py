import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearnerModule(nn.Module):
    def __init__(self, num_tokens, use_sum_pooling=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.use_sum_pooling = use_sum_pooling
        self.conv_layers = nn.Sequential(
            nn.LayerNorm(1),
            nn.Conv2d(1, num_tokens, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        bs, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # Change to [bs, c, h, w]
        attn = self.conv_layers(x)  # [bs, num_tokens, h, w]
        attn = torch.sigmoid(attn).view(
            bs, self.num_tokens, -1, 1
        )  # [bs, num_tokens, hw, 1]

        x = x.view(bs, c, -1).unsqueeze(1)  # [bs, 1, hw, c]
        x = (
            torch.sum(x * attn, dim=2)
            if self.use_sum_pooling
            else torch.mean(x * attn, dim=2)
        )
        return x


class TokenLearnerModuleV11(nn.Module):
    def __init__(self, num_tokens, bottleneck_dim=64, dropout_rate=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, num_tokens),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        bs, h, w, c = x.shape
        x = x.view(bs, h * w, c)
        attn = self.mlp(x)  # [bs, h*w, num_tokens]
        attn = F.softmax(attn, dim=-1)
        x = torch.einsum("bsc,bch->bsh", attn, x)  # [bs, num_tokens, c]
        return x


class TokenFuser(nn.Module):
    def __init__(self, bottleneck_dim=64, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, original):
        x = self.norm1(x).permute(0, 2, 1)
        x = self.norm2(nn.Linear(original.shape[1], x.shape[1])(x)).permute(0, 2, 1)
        attn = torch.sigmoid(self.mlp(original))
        x = torch.einsum("bsc,bsh->bsh", x, attn)
        return self.dropout(x) + original


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.attn(x, x, x)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.mlp(x))
        return self.norm2(x)


class TokenLearnerViT(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim,
        num_layers,
        num_heads,
        mlp_dim,
        num_tokens,
        use_v11=True,
    ):
        super().__init__()
        self.embedding = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.token_learner = (
            TokenLearnerModuleV11(num_tokens)
            if use_v11
            else TokenLearnerModule(num_tokens)
        )
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).flatten(2).permute(0, 2, 1)
        x = self.token_learner(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class TokenLearnerModuleV12(nn.Module):
    def __init__(
        self, in_num_tokens=128, out_tokens=32, bottleneck_dim=64, dropout_rate=0.0
    ):
        """
        A spatio-temporal token selection module that learns to select informative tokens across frames.

        Args:
            num_tokens (int): Number of tokens to learn across space-time.
            bottleneck_dim (int): Dimension of the intermediate MLP layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_tokens = in_num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_num_tokens, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_tokens),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, frames, height, width, embedding_dim]

        Returns:
            Spatio-temporal tokens of shape [batch_size, num_tokens, embedding_dim]
        """

        # import pdb; pdb.set_trace()
        bs, f, tokens, c = x.shape

        # Reshape input into space-time tokens
        x = x.view(bs, f * tokens, c)  # [bs, T=N×H×W, c] where T is space-time tokens

        # Compute selection weights (soft-attention)
        attn = self.mlp(x)  # [bs, T, num_tokens]
        attn = F.softmax(attn, dim=-1)  # Normalize over num_tokens

        # Select informative tokens using weighted sum
        selected_tokens = torch.einsum("btc,btm->bmc", x, attn)  # [bs, num_tokens, c]

        return selected_tokens


class TokenProcessor(nn.Module):
    def __init__(self, out_tokens=32, embedding_dim=256, conv_out_channels=128):
        """
        Token Processor to further compress selected tokens into a fixed-size embedding.

        Args:
            out_tokens (int): Number of tokens after selection by TokenLearnerModule.
            embedding_dim (int): Feature dimension per token.
            conv_out_channels (int): Output channels of the convolution layers.
        """
        super().__init__()
        self.out_tokens = out_tokens

        # Convolution layers to process tokens
        self.conv1 = nn.Conv1d(
            embedding_dim, conv_out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False
        )

        # Fully connected layer: Adjust input size dynamically
        self.fc_in_dim = conv_out_channels * out_tokens
        self.fc = nn.Linear(self.fc_in_dim, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, out_tokens, embedding_dim]
        Returns:
            Compressed embedding: [batch_size, embedding_dim]
        """
        bs, t, c = x.shape  # [batch_size, out_tokens, embedding_dim]

        # Reshape for Conv1D: Convert to [batch, embedding_dim, out_tokens]
        x = x.permute(0, 2, 1)  # [batch, embedding_dim, out_tokens]

        # Apply convolutional layers
        x = self.conv1(x)  # [batch, conv_out_channels, out_tokens]
        x = F.gelu(x)
        x = self.conv2(x)  # [batch, conv_out_channels, out_tokens]

        # Flatten last two dimensions
        x = x.view(bs, -1)  # [batch, conv_out_channels * out_tokens]

        # Fully connected layer to compress into final embedding
        x = self.fc(x)  # [batch, embedding_dim]

        return x


# class TokenProcessor(nn.Module):
#     def __init__(self, num_tokens=16, frames=4, embedding_dim=256, conv_out_channels=128):
#         super().__init__()
#         self.num_tokens = num_tokens
#         self.conv1 = nn.Conv1d(embedding_dim, conv_out_channels, kernel_size=3, padding=1, bias=False)  # Fix input channels
#         self.conv2 = nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False)

#         # Fully connected layer: Compute correct input size dynamically
#         self.fc_in_dim = conv_out_channels * num_tokens * frames # Compute dynamically
#         self.fc = nn.Linear(self.fc_in_dim, embedding_dim)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape [batch, frames, num_tokens, embedding_dim]
#         Returns:
#             Compressed embedding: [batch, embedding_dim]
#         """
#         bs, f, t, c = x.shape  # (batch, frames, num_tokens, embedding)

#         # import pdb; pdb.set_trace()
#         # Reshape for Conv1D: [batch * frames, num_tokens, embedding]
#         x = x.view(bs * f, t, c).permute(0, 2, 1)  # Fix: [bs*frames, embedding, num_tokens]

#         # Apply convolutional layers
#         x = self.conv1(x)  # [bs*frames, conv_out_channels, num_tokens]
#         x = F.gelu(x)
#         x = self.conv2(x)  # [bs*frames, conv_out_channels, num_tokens]

#         # Reshape back to batch-wise format
#         x = x.view(bs, f, -1, t)  # [bs, frames, conv_out_channels, num_tokens]

#         # Flatten last two dimensions
#         x = x.view(bs, -1)  # [bs, frames * conv_out_channels * num_tokens]

#         # Fully connected layer to embedding dimension
#         x = self.fc(x)  # [bs, embedding_dim]

#         return x

# class TokenProcessor(nn.Module):
#     def __init__(self, num_tokens=16, frames=4, embedding_dim=256, conv_out_channels=128):
#         super().__init__()
#         self.num_tokens = num_tokens

#         # Convolutional layers with residual connections
#         self.conv1 = nn.Conv1d(embedding_dim, conv_out_channels, kernel_size=3, padding=1, bias=False)
#         self.conv2 = nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False)
#         self.conv3 = nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False)
#         self.conv4 = nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False)

#         # Projection layer to match dimensions for residual connection
#         self.shortcut = nn.Conv1d(embedding_dim, conv_out_channels, kernel_size=1, bias=False)

#         # Fully connected layers with residual connections
#         self.fc_in_dim = conv_out_channels * num_tokens * frames
#         self.fc1 = nn.Linear(self.fc_in_dim, embedding_dim)
#         self.fc2 = nn.Linear(embedding_dim, embedding_dim)

#         # Shortcut for FC residual connection
#         self.fc_shortcut = nn.Linear(self.fc_in_dim, embedding_dim)  # Added projection layer

#         # BatchNorm for stability
#         self.bn1 = nn.BatchNorm1d(conv_out_channels)
#         self.bn2 = nn.BatchNorm1d(conv_out_channels)
#         self.bn3 = nn.BatchNorm1d(conv_out_channels)
#         self.bn4 = nn.BatchNorm1d(conv_out_channels)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape [batch, frames, num_tokens, embedding_dim]
#         Returns:
#             Compressed embedding: [batch, embedding_dim]
#         """
#         bs, f, t, c = x.shape  # (batch, frames, num_tokens, embedding)

#         # Reshape for Conv1D: [batch * frames, num_tokens, embedding] -> [batch * frames, embedding, num_tokens]
#         x = x.view(bs * f, t, c).permute(0, 2, 1)

#         # Apply shortcut for residual connection
#         res = self.shortcut(x)
#         x = F.gelu(self.bn1(self.conv1(x)))
#         x = F.gelu(self.bn2(self.conv2(x))) + res  # First residual connection

#         res = x
#         x = F.gelu(self.bn3(self.conv3(x)))
#         x = F.gelu(self.bn4(self.conv4(x))) + res  # Second residual connection

#         # Reshape back to batch-wise format
#         x = x.view(bs, f, -1, t)  # [bs, frames, conv_out_channels, num_tokens]

#         # Flatten last two dimensions
#         x = x.view(bs, -1)  # [bs, frames * conv_out_channels * num_tokens]

#         # Fully connected layers with residual connection
#         res = self.fc_shortcut(x)  # Project residual to embedding_dim
#         x = F.gelu(self.fc1(x))
#         x = self.fc2(x) + res  # Residual connection at the linear layers

#         return x
