
# Source: Adapted and simplified from "An Image is Worth 16x16 Words"
# Based on: https://github.com/lucidrains/vit-pytorch (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=256, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # shape: (B, emb_size, H/P, W/P)
        x = x.flatten(2)            # shape: (B, emb_size, N)
        x = x.transpose(1, 2)       # shape: (B, N, emb_size)
        return x


class YourVIT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=4,
        emb_size=256,
        img_size=32,
        num_classes=10,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        n_patches = self.patch_embedding.n_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)               # (B, N, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)     # (B, N+1, emb_size)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.encoder(x)                       # (B, N+1, emb_size)
        cls_output = x[:, 0]                      # take CLS token
        return self.mlp_head(cls_output)
