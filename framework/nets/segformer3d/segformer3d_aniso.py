"""SegFormer3D with anisotropic (non-cubic) input support.

Changes vs upstream architectures/segformer3d.py (OSUPCVLab/SegFormer3D):
  1. Removed `@torch.jit.script cube_root` (fails under torch>=2.x: round() typed float).
  2. Replaced every `n = cube_root(N); x.reshape(B,n,n,n,C)` with explicit (D,H,W)
     threaded through the forward — so non-cubic patches like [64,160,160] work.
Attention / PatchEmbedding convs / SR convs / all-MLP decoder are UNCHANGED.
SR-conv divisibility for [64,160,160] checked: stage1 (16,40,40)/sr4, stage2 (8,20,20)/sr2 hold.
"""
import torch
import math
from torch import nn
from functools import partial
from typing import Tuple, List


class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [4, 4, 4, 4],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels, sr_ratios=sr_ratios, embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size, patch_stride=patch_stride,
            patch_padding=patch_padding, mlp_ratios=mlp_ratios, num_heads=num_heads,
            depths=depths,
        )
        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes, dropout=decoder_dropout,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            k = m.kernel_size
            fan_out = (k[0] * k[1] * (k[2] if len(k) == 3 else 1)) * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.segformer_encoder(x)
        c1, c2, c3, c4 = x[0], x[1], x[2], x[3]
        x = self.segformer_decoder(c1, c2, c3, c4)
        return x


class PatchEmbedding(nn.Module):
    """Returns (patches [B,N,C], D, H, W). Tracks spatial shape explicitly."""
    def __init__(self, in_channel=4, embed_dim=768, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        patches = self.patch_embeddings(x)          # (B, C, D, H, W)
        D, H, W = patches.shape[-3:]
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, C)
        patches = self.norm(patches)
        return patches, D, H, W


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, sr_ratio=2, qkv_bias=False,
                 attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_head_dim = embed_dim // num_heads
        self.scale = self.attention_head_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout_p = attn_dropout
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, shape) -> torch.Tensor:
        D, H, W = shape
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.attention_head_dim) \
                           .permute(0, 2, 1, 3).contiguous()
        if self.sr_ratio > 1:
            # reshape using explicit (D,H,W) instead of cube_root
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W).contiguous()
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.sr_norm(x_)
            kv = self.key_value(x_).reshape(B, -1, 2, self.num_heads, self.attention_head_dim) \
                                       .permute(2, 0, 3, 1, 4)
        else:
            kv = self.key_value(x).reshape(B, -1, 2, self.num_heads, self.attention_head_dim) \
                                     .permute(2, 0, 3, 1, 4)
        k, v = kv[0].contiguous(), kv[1].contiguous()
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C).contiguous()
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=2, num_heads=8, sr_ratio=2,
                 qkv_bias=False, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads,
            sr_ratio=sr_ratio, qkv_bias=qkv_bias, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x, shape):
        x = x + self.attention(self.norm1(x), shape)
        x = x + self.mlp(self.norm2(x), shape)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, in_channels=4, sr_ratios=[8, 4, 2, 1], embed_dims=[64, 128, 320, 512],
                 patch_kernel_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_padding=[3, 1, 1, 1],
                 mlp_ratios=[2, 2, 2, 2], num_heads=[1, 2, 5, 8], depths=[2, 2, 2, 2]):
        super().__init__()
        self.embed_1 = PatchEmbedding(in_channel=in_channels, embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0], stride=patch_stride[0], padding=patch_padding[0])
        self.embed_2 = PatchEmbedding(in_channel=embed_dims[0], embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1], stride=patch_stride[1], padding=patch_padding[1])
        self.embed_3 = PatchEmbedding(in_channel=embed_dims[1], embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2], stride=patch_stride[2], padding=patch_padding[2])
        self.embed_4 = PatchEmbedding(in_channel=embed_dims[2], embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3], stride=patch_stride[3], padding=patch_padding[3])
        self.tf_block1 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[0], num_heads=num_heads[0],
            mlp_ratio=mlp_ratios[0], sr_ratio=sr_ratios[0], qkv_bias=True) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.tf_block2 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[1], num_heads=num_heads[1],
            mlp_ratio=mlp_ratios[1], sr_ratio=sr_ratios[1], qkv_bias=True) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.tf_block3 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[2], num_heads=num_heads[2],
            mlp_ratio=mlp_ratios[2], sr_ratio=sr_ratios[2], qkv_bias=True) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.tf_block4 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[3], num_heads=num_heads[3],
            mlp_ratio=mlp_ratios[3], sr_ratio=sr_ratios[3], qkv_bias=True) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        out = []
        x, D, H, W = self.embed_1(x)
        shape = (D, H, W)
        B = x.shape[0]
        for blk in self.tf_block1:
            x = blk(x, shape)
        x = self.norm1(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        x, D, H, W = self.embed_2(x)
        shape = (D, H, W)
        for blk in self.tf_block2:
            x = blk(x, shape)
        x = self.norm2(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        x, D, H, W = self.embed_3(x)
        shape = (D, H, W)
        for blk in self.tf_block3:
            x = blk(x, shape)
        x = self.norm3(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        x, D, H, W = self.embed_4(x)
        shape = (D, H, W)
        for blk in self.tf_block4:
            x = blk(x, shape)
        x = self.norm4(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        return out


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shape) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, shape)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    """Uses explicit (D,H,W) instead of cube_root."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x: torch.Tensor, shape) -> torch.Tensor:
        D, H, W = shape
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, D, H, W).contiguous()
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class MLP_(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.bn(x)
        return x


class SegFormerDecoderHead(nn.Module):
    """Unchanged from upstream — uses interpolate, no cube_root."""
    def __init__(self, input_feature_dims=[512, 320, 128, 64], decoder_head_embedding_dim=256,
                 num_classes=3, dropout=0.0):
        super().__init__()
        self.linear_c4 = MLP_(input_dim=input_feature_dims[0], embed_dim=decoder_head_embedding_dim)
        self.linear_c3 = MLP_(input_dim=input_feature_dims[1], embed_dim=decoder_head_embedding_dim)
        self.linear_c2 = MLP_(input_dim=input_feature_dims[2], embed_dim=decoder_head_embedding_dim)
        self.linear_c1 = MLP_(input_dim=input_feature_dims[3], embed_dim=decoder_head_embedding_dim)
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(4 * decoder_head_embedding_dim, decoder_head_embedding_dim, 1, 1, bias=False),
            nn.BatchNorm3d(decoder_head_embedding_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.linear_pred = nn.Conv3d(decoder_head_embedding_dim, num_classes, 1)
        self.upsample_volume = nn.Upsample(scale_factor=4.0, mode="trilinear", align_corners=False)

    def forward(self, c1, c2, c3, c4):
        n = c4.shape[0]
        _c4 = self.linear_c4(c4).permute(0, 2, 1) \
                .reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4]).contiguous()
        _c4 = torch.nn.functional.interpolate(_c4, size=c1.size()[2:], mode="trilinear", align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1) \
                .reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4]).contiguous()
        _c3 = torch.nn.functional.interpolate(_c3, size=c1.size()[2:], mode="trilinear", align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1) \
                .reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4]).contiguous()
        _c2 = torch.nn.functional.interpolate(_c2, size=c1.size()[2:], mode="trilinear", align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1) \
                .reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4]).contiguous()
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.upsample_volume(x)
        return x
