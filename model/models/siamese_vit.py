# Siamese Vision Transformer for Tactile Sensing
# Each image (deformed/undeformed) is encoded with a shared ViT encoder,
# then features are fused and decoded to produce the output.

from functools import partial
from typing import List, Literal

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class SiameseViT(nn.Module):
    """
    Siamese Vision Transformer for dense prediction from paired images.

    Architecture:
        1. Shared ViT encoder processes each image (deformed, undeformed) separately
        2. Features are fused (concat, diff, or both)
        3. ViT decoder produces dense output

    Input: (B, 6, H, W) where channels 0-2 are deformed, 3-5 are undeformed
    Output: (B, out_chans, H, W) dense prediction
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,  # per image (deformed or undeformed)
        out_chans: List[int] = [3, 3, 3, 3, 3],
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.,
        norm_layer: nn.Module = nn.LayerNorm,
        fusion_method: Literal["concat", "diff", "concat_diff"] = "concat_diff",
    ):
        """
        Args:
            img_size: Input image size (square)
            patch_size: Patch size for ViT
            in_chans: Input channels per image (3 for RGB)
            out_chans: List of output channel counts per head
            embed_dim: Encoder embedding dimension
            depth: Number of encoder transformer blocks
            num_heads: Number of attention heads in encoder
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            mlp_ratio: MLP hidden dim ratio
            norm_layer: Normalization layer
            fusion_method: How to fuse encoder features
                - "concat": Concatenate deformed and undeformed features
                - "diff": Compute difference (deformed - undeformed)
                - "concat_diff": Concatenate both images' features AND their difference
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method

        # Compute total output channels
        if isinstance(out_chans, list):
            self.out_chans = sum(out_chans)
            self.out_chans_list = out_chans
        else:
            self.out_chans = out_chans
            self.out_chans_list = [out_chans]

        # --------------------------------------------------------------------------
        # Shared encoder (processes each image separately with same weights)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Fusion layer - projects fused features to decoder dimension
        if fusion_method == "concat":
            fusion_input_dim = embed_dim * 2  # deformed + undeformed
        elif fusion_method == "diff":
            fusion_input_dim = embed_dim  # just difference
        elif fusion_method == "concat_diff":
            fusion_input_dim = embed_dim * 3  # deformed + undeformed + diff
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion_proj = nn.Linear(fusion_input_dim, decoder_embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Final projection: predict pixel values for each patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * self.out_chans,
            bias=True
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch embedding like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_single_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image through the shared encoder.

        Args:
            x: (B, 3, H, W) single image

        Returns:
            (B, num_patches + 1, embed_dim) encoded features with cls token
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.encoder_norm(x)

        return x

    def fuse_features(
        self,
        feat_deformed: torch.Tensor,
        feat_undeformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features from deformed and undeformed images.

        Args:
            feat_deformed: (B, L, D) features from deformed image
            feat_undeformed: (B, L, D) features from undeformed image

        Returns:
            (B, L, fusion_dim) fused features
        """
        if self.fusion_method == "concat":
            fused = torch.cat([feat_deformed, feat_undeformed], dim=-1)
        elif self.fusion_method == "diff":
            fused = feat_deformed - feat_undeformed
        elif self.fusion_method == "concat_diff":
            diff = feat_deformed - feat_undeformed
            fused = torch.cat([feat_deformed, feat_undeformed, diff], dim=-1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode fused features to dense prediction.

        Args:
            x: (B, L, decoder_embed_dim) fused and projected features

        Returns:
            (B, L-1, patch_size^2 * out_chans) per-patch predictions (excluding cls)
        """
        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # Project to pixel predictions
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from patch predictions.

        Args:
            x: (B, num_patches, patch_size^2 * out_chans)

        Returns:
            (B, out_chans, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], f"Expected square grid, got {x.shape[1]} patches"

        x = x.reshape(x.shape[0], h, w, p, p, self.out_chans)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.shape[0], self.out_chans, h * p, w * p)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 6, H, W) input with deformed (0:3) and undeformed (3:6) images

        Returns:
            (B, out_chans, H, W) dense prediction
        """
        # Split input into deformed and undeformed images
        deformed = x[:, :3, :, :]      # (B, 3, H, W)
        undeformed = x[:, 3:6, :, :]   # (B, 3, H, W)

        # Encode each image with shared encoder
        feat_deformed = self.encode_single_image(deformed)      # (B, L, D)
        feat_undeformed = self.encode_single_image(undeformed)  # (B, L, D)

        # Fuse features
        fused = self.fuse_features(feat_deformed, feat_undeformed)  # (B, L, fusion_dim)

        # Project to decoder dimension
        fused = self.fusion_proj(fused)  # (B, L, decoder_embed_dim)

        # Decode to patch predictions
        pred = self.decode(fused)  # (B, num_patches, patch_size^2 * out_chans)

        # Reconstruct image
        pred = self.unpatchify(pred)  # (B, out_chans, H, W)

        return pred

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder only (for feature extraction).

        Returns fused features before decoding.
        """
        deformed = x[:, :3, :, :]
        undeformed = x[:, 3:6, :, :]

        feat_deformed = self.encode_single_image(deformed)
        feat_undeformed = self.encode_single_image(undeformed)

        fused = self.fuse_features(feat_deformed, feat_undeformed)
        fused = self.fusion_proj(fused)

        return fused

    def get_encoder_features(self, x: torch.Tensor) -> tuple:
        """
        Get individual encoder features for each image.

        Returns:
            (feat_deformed, feat_undeformed) each of shape (B, L, D)
        """
        deformed = x[:, :3, :, :]
        undeformed = x[:, 3:6, :, :]

        feat_deformed = self.encode_single_image(deformed)
        feat_undeformed = self.encode_single_image(undeformed)

        return feat_deformed, feat_undeformed


# Factory functions for different model sizes

def siamese_vit_base_patch16(**kwargs):
    """Base model: 12 layers, 768 dim, 12 heads"""
    model = SiameseViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def siamese_vit_large_patch16(**kwargs):
    """Large model: 24 layers, 1024 dim, 16 heads"""
    model = SiameseViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def siamese_vit_small_patch16(**kwargs):
    """Small model: 6 layers, 384 dim, 6 heads"""
    model = SiameseViT(
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
