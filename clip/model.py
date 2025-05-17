from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip.myAtt as myAtt

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.ops.modules import MSDeformAttn
from torch.nn.init import normal_
from clip.comer_modules import CNN, CTIBlock, deform_inputs, CTI_toV, CTI_toC, MRFP, deform_inputs_only_one



def upsample_pos_emb(emb, new_size):
    # upsample the pretrained embedding for higher resolution
    # emb size NxD
    first = emb[:1, :]
    emb = emb[1:, :]
    N, D = emb.size(0), emb.size(1)
    size = int(np.sqrt(N))
    assert size * size == N
    #new_size = size * self.upsample
    emb = emb.permute(1, 0)
    emb = emb.view(1, D, size, size).contiguous()
    emb = F.upsample(emb, size=new_size, mode='bilinear',)
    emb = emb.view(D, -1).contiguous()
    emb = emb.permute(1, 0)
    emb = torch.cat([first, emb], 0)
    emb = nn.parameter.Parameter(emb.half())
    return emb

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, H, W):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (H//32,W//32))
        x = x + self.positional_embedding_new[:, None, :].to(x.dtype)  # (HW+1)NC
        x, attn_weight = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, H, W):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)#(1,,2048, 7, 7)
        x_pooled = self.attnpool(x, H, W)

        return x_pooled


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = myAtt.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)#[0]

    def forward(self, x: torch.Tensor):
        attn_output, attn_weight = self.attention(self.ln_1(x))#(L,N,E)  (N,L,L)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weight



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, require_all_fts=False):
        attn_weights = []
        x_all = []
        with torch.no_grad():
            layers = self.layers if x.shape[0] == 77 else self.layers-1
            for i in range(layers):
                x, attn_weight = self.resblocks[i](x)
                x_all.append(x)
                attn_weights.append(attn_weight)
        '''
        for i in range(self.layers-1, self.layers):
            x, attn_weight = self.resblocks[i](x)
            attn_weights.append(attn_weight)
            #feature_map_list.append(x)
        '''
        if require_all_fts == True:
            return x_all, attn_weights
        else:
            return x, attn_weights


# class VisionTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)

#         self.transformer = Transformer(width, layers, heads)

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
#         self.patch_size = patch_size

#     def forward(self, x: torch.Tensor, H, W, require_all_fts=False):

#         self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (H//16,W//16))
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding_new.to(x.dtype)
#         x = self.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x, attn_weight = self.transformer(x, require_all_fts=require_all_fts)


#         '''
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         x = self.ln_post(x)

#         if self.proj is not None:
#             x = x @ self.proj
#         '''

#         return x, attn_weight#cls_attn

class Adapter(nn.Module):
    """Adapter module for ViT"""
    def __init__(self, dim, down_ratio=4):
        super().__init__()
        # Adapter 구현
        self.down_proj = nn.Linear(dim, dim // down_ratio)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(dim // down_ratio, dim)
        self.layer_norm = LayerNorm(dim)
        
    def forward(self, x):
        shortcut = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return shortcut + x
    
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        # 기존 코드 유지 (일부 생략)
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # ViT-B/16은 총 12개의 transformer block을 가짐 (11개는 정상 forward, 12번째는 Grad-CAM용)
        self.transformer = Transformer(width, layers, heads)
        self.patch_size = patch_size
        
        # ViT-Comer에서 추가된 부분
        self.pretrain_size = (input_resolution, input_resolution)
        
        # Stage 구분 (총 11개의 block을 4개 stage로 나눔)
        self.stage_indices = [
            [0, 1],  # Stage 1: 블록 0-1 (2개)
            [2, 4],  # Stage 2: 블록 2-4 (3개)
            [5, 7],  # Stage 3: 블록 5-7 (3개)
            [8, 10]  # Stage 4: 블록 8-10 (3개)
        ]
        
        # CNN backbone 추가
        self.spm = CNN(inplanes=64, embed_dim=width)
        self.level_embed = nn.Parameter(torch.zeros(3, width))
        
        # CTI_toV 모듈 (CNN -> ViT 방향의 특징 융합)
        self.cti_to_v_modules = nn.ModuleList([
            CTI_toV(dim=width, num_heads=heads//2, n_points=4,
                   deform_ratio=1.0, init_values=0.,
                   norm_layer=LayerNorm, drop=0.1, drop_path=0.1,
                   cffn_ratio=0.25)
            for _ in range(len(self.stage_indices))
        ])
        
        # CTI_toC 모듈 (ViT -> CNN 방향의 특징 융합)
        self.cti_to_c_modules = nn.ModuleList([
            CTI_toC(dim=width, num_heads=heads//2, n_points=4,
                   deform_ratio=1.0, with_cffn=True,
                   cffn_ratio=0.25, drop=0.1, drop_path=0.1,
                   norm_layer=LayerNorm, cnn_feature_interaction=True)
            for _ in range(len(self.stage_indices))
        ])
        
        # MRFP 모듈 (Multi-Resolution Feature Processing)
        self.mrfp_modules = nn.ModuleList([
            MRFP(width, hidden_features=int(width * 6.0))
            for _ in range(len(self.stage_indices))
        ])
        
        # Adapter 모듈 추가 (각 CTI에 대응)
        self.adapters_to_v = nn.ModuleList([
            Adapter(width) for _ in range(len(self.stage_indices))
        ])
 
        self.adapters_to_c = nn.ModuleList([
            Adapter(width) for _ in range(len(self.stage_indices))
        ])


        self.adapters = nn.ModuleList([
    Adapter(width) for _ in range(layers)
])       
        # interactions
        self.interactions = nn.ModuleList([
    CTIBlock(dim=width, num_heads=heads//2, n_points=4,
             init_values=0., drop_path=0.1,
             norm_layer=LayerNorm, with_cffn=True,
             cffn_ratio=0.25, deform_ratio=1.0,
             use_CTI_toV=True, use_CTI_toC=True,
             dim_ratio=6.0,
             cnn_feature_interaction=True,
             extra_CTI=(i == len(self.stage_indices) - 1))
    for i in range(len(self.stage_indices))
])

        # 최종 출력을 위한 normalization layers
        self.norm1 = nn.BatchNorm2d(width)
        self.norm2 = nn.BatchNorm2d(width)
        self.norm3 = nn.BatchNorm2d(width)
        self.norm4 = nn.BatchNorm2d(width)
        
        # 업샘플링 레이어
        self.up = nn.ConvTranspose2d(width, width, 2, 2)
        
        # 8개의 CTI 출력을 channel-wise concat 후 적용할 1x1 convolution
        self.final_conv = nn.Conv2d(width * 8, width, kernel_size=1)
        
        # 초기화
        self._init_weights()

            
    def _init_weights(self):
        # SPM, MRFP, CTI_toV, CTI_toC 등의 가중치 초기화
        self.spm.apply(self._init_weights_fn)
        self.mrfp_modules.apply(self._init_weights_fn)
        self.cti_to_v_modules.apply(self._init_weights_fn)
        self.cti_to_c_modules.apply(self._init_weights_fn)
        self.up.apply(self._init_weights_fn)
        nn.init.normal_(self.level_embed, std=0.02)
        
        # 필요한 경우 adapters도 초기화
        self.adapters_to_v.apply(self._init_weights_fn)
        self.adapters_to_c.apply(self._init_weights_fn)
        
    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    
    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    # def _get_pos_embed(self, pos_embed, H, W):
    #     pos_embed = pos_embed.reshape(
    #         1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
    #     pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
    #         reshape(1, -1, H * W).permute(0, 2, 1)
    #     return pos_embed

    def _get_pos_embed(self, pos_embed, H, W):
        # 원래 구현에서는 self.pretrain_size를 사용하여 reshape 크기를 계산했지만,
        # 실제 positional embedding의 크기를 기반으로 계산하도록 수정
        
        # positional embedding의 크기 확인
        N = pos_embed.shape[0]  # 전체 토큰 수 (class token 포함)
        
        # class token이 있는 경우 제외
        class_pos_embed = pos_embed[:1]
        pos_embed = pos_embed[1:]
        
        # 남은 positional embedding의 크기로부터 원래 grid 크기 계산
        # grid_size * grid_size = pos_embed.shape[0]
        grid_size = int(pos_embed.shape[0] ** 0.5)
        
        # 원래 형태로 reshape
        pos_embed = pos_embed.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
        
        # 새로운 크기로 interpolate
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1)
        
        # class token의 positional embedding 다시 추가
        pos_embed = torch.cat([class_pos_embed.unsqueeze(0), pos_embed], dim=1)
        
        return pos_embed
    
    def forward(self, x: torch.Tensor, H, W, require_all_fts=False):
        # deform_inputs 준비
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        
        # CNN backbone (SPM) 처리
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        
        # ViT branch 초기화
        x_vit = self.conv1(x)
        x_vit = x_vit.reshape(x_vit.shape[0], x_vit.shape[1], -1)
        x_vit = x_vit.permute(0, 2, 1)
        x_vit = torch.cat([self.class_embedding.to(x_vit.dtype) + torch.zeros(x_vit.shape[0], 1, x_vit.shape[-1], dtype=x_vit.dtype, device=x_vit.device), x_vit], dim=1)
        
        # Position embedding 적용
        pos_embed = self._get_pos_embed(self.positional_embedding, H//16, W//16)
        x_vit = x_vit + pos_embed.to(x_vit.dtype)
        x_vit = self.ln_pre(x_vit)
        x_vit = x_vit.permute(1, 0, 2)  # NLD -> LND
        
        bs, _, dim = x_vit.shape[1], x_vit.shape[0], x_vit.shape[2]
        
        # 각 CTI 모듈의 출력을 저장할 리스트
        cti_outputs = []
        
        # 현재 ViT 및 CNN feature
        current_vit = x_vit
        current_cnn = c
        
        # 각 stage 별로 처리
        for stage_idx, (start_block, end_block) in enumerate(self.stage_indices):
            # MRFP 적용 (CNN feature 처리)
            processed_cnn = self.mrfp_modules[stage_idx](current_cnn.to(self.mrfp_modules[stage_idx].fc1.weight.dtype), H//16, W//16)
            
            # Stage 시작 전 adapter 적용 (ViT feature에)
            adapter_v_output = self.adapters_to_v[stage_idx](current_vit)
            
            # CTI_toV 적용 (CNN -> ViT)
            deform_input = deform_inputs_only_one(adapter_v_output, H*16, W*16)
            cti_v_output = self.cti_to_v_modules[stage_idx](
                query=adapter_v_output,
                reference_points=deform_input[0],
                feat=processed_cnn,
                spatial_shapes=deform_input[1],
                level_start_index=deform_input[2],
                H=H//16, W=W//16
            )
            
            # CTI_toV 출력 저장
            cti_outputs.append(cti_v_output)
            
            # Residual connection: current_vit + cti_v_output (element-wise sum)
            stage_input = current_vit + cti_v_output
            
            # Stage 내의 transformer blocks 처리
            stage_output = stage_input
            for block_idx in range(start_block, end_block + 1):
                # Transformer block 적용
                block_output, _ = self.transformer.resblocks[block_idx](stage_output)
                stage_output = block_output
            
            # Stage 종료 후 adapter 적용 (ViT feature에)
            adapter_c_output = self.adapters_to_c[stage_idx](stage_output)
            
            # CTI_toC 적용 (ViT -> CNN)
            cti_c_output = self.cti_to_c_modules[stage_idx](
                query=current_cnn,
                reference_points=deform_inputs2[0],
                feat=adapter_c_output,
                spatial_shapes=deform_inputs2[1],
                level_start_index=deform_inputs2[2],
                H=H//16, W=W//16
            )
            
            # CTI_toC 출력 저장
            cti_outputs.append(cti_c_output)
            
            # Residual connection: current_cnn + cti_c_output (element-wise sum)
            current_cnn = current_cnn + cti_c_output
            
            # 다음 stage를 위해 현재 ViT feature 업데이트
            current_vit = stage_output
        
        # CNN feature 분리 및 reshape - 필요시 사용 가능하도록 유지
        c2_len = c2.size(1)
        c3_len = c3.size(1)
        
        c2_new = current_cnn[:, 0:c2_len, :]
        c3_new = current_cnn[:, c2_len:c2_len + c3_len, :]
        c4_new = current_cnn[:, c2_len + c3_len:, :]
        
        c2_new = c2_new.transpose(1, 2).view(bs, dim, H//8, W//8).contiguous()
        c3_new = c3_new.transpose(1, 2).view(bs, dim, H//16, W//16).contiguous()
        c4_new = c4_new.transpose(1, 2).view(bs, dim, H//32, W//32).contiguous()
        c1_new = self.up(c2_new) + c1
        
        # Final Norm
        f1 = self.norm1(c1_new)
        f2 = self.norm2(c2_new)
        f3 = self.norm3(c3_new)
        f4 = self.norm4(c4_new)
        
        # 8개의 CTI 출력을 channel-wise concat하고 1x1 convolution 적용
        unified_cti_outputs = []
        
        for cti_output in cti_outputs:
            # CTI 출력 형태 변환 (LND -> NCHW 또는 NLD -> NCHW)
            if cti_output.shape[0] != bs:  # LND 형태인 경우
                cti_output = cti_output.permute(1, 0, 2)  # LND -> NLD
            
            # 공간 차원으로 변환
            if len(cti_output.shape) == 3:  # NLD 형태
                if cti_output.shape[1] > 1:  # class token 등이 포함된 경우
                    # class token 제거하고 공간 차원으로 변환
                    seq_len = cti_output.shape[1]
                    if seq_len == (H//16)*(W//16) + 1:  # class token이 있는 경우
                        cti_output = cti_output[:, 1:, :]  # class token 제거
                    
                    # 공간 차원으로 reshape
                    h_dim = w_dim = int(cti_output.shape[1] ** 0.5)
                    cti_output = cti_output.reshape(bs, h_dim, w_dim, -1).permute(0, 3, 1, 2)
                else:
                    # 단일 token인 경우 (예: global feature)
                    cti_output = cti_output.unsqueeze(-1).unsqueeze(-1)
            
            # 모든 feature를 H//16 x W//16 크기로 통일
            if cti_output.shape[2:] != (H//16, W//16):
                cti_output = F.interpolate(cti_output, size=(H//16, W//16), mode='bilinear', align_corners=False)
            
            unified_cti_outputs.append(cti_output)
        
        # Channel-wise concat 후 1x1 convolution 적용
        concat_cti = torch.cat(unified_cti_outputs, dim=1)  # N x (8*C) x H//16 x W//16
        final_cti = self.final_conv(concat_cti)  # N x C x H//16 x W//16
        
        if require_all_fts:
            # 11번째 transformer block의 output은 Grad-CAM을 위해 저장
            last_vit_output = current_vit
            
            # CTI outputs만을 출력으로 사용
            return last_vit_output, [f1, f2, f3, f4], final_cti
        else:
            # 기존 VisionTransformer의 출력 형태 유지
            x_vit = current_vit.permute(1, 0, 2)  # LND -> NLD
            
            if not hasattr(self, 'ln_post'):
                self.ln_post = LayerNorm(dim)
            x_vit = self.ln_post(x_vit[:, 0])
            
            if not hasattr(self, 'proj'):
                self.proj = nn.Parameter(scale * torch.randn(dim, self.output_dim))
            if self.proj is not None:
                x_vit = x_vit @ self.proj
            
            return x_vit, None
        
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # def encode_image(self, image, H, W, require_all_fts=False):
    #     f_x, f_attn = self.visual(image.type(self.dtype), H, W, require_all_fts=require_all_fts)
    #     # f = self.visual(image.type(self.dtype), H, W, require_all_fts=require_all_fts)
    #     return f_x, f_attn

    def encode_image(self, image, H, W, require_all_fts=False):
        """
        ViT-Comer가 적용된 encode_image 메서드
        """
        if require_all_fts:
            # ViT-Comer 모드
            last_vit_output, multi_level_features, final_cti = self.visual(image.type(self.dtype), H, W, require_all_fts=True)
            
            # last_vit_output: 마지막 transformer block의 출력 (LND 형태)
            # multi_level_features: [f1, f2, f3, f4] 4개의 스케일 특징맵
            # final_cti: 8개의 CTI 출력을 channel-wise concat 후 1x1 conv 적용한 결과
            
            # Grad-CAM에 사용될 transformer block 출력 반환
            return last_vit_output, final_cti
        else:
            # 기존 방식
            f_x, f_attn = self.visual(image.type(self.dtype), H, W, require_all_fts=False)
            return f_x, f_attn
    
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_weight = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward_last_layer(self, image_features, text_features):
        x, attn_weight = self.visual.transformer.resblocks[self.visual.transformer.layers-1](image_features)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)
        x = torch.mean(x[:, 1:, :], dim=1)

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        image_features = x

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits_per_image = logits_per_image.softmax(dim=-1)

        return logits_per_image, attn_weight

    def forward(self, image, text):
        image_features, feature_map, cls_attn = self.encode_image(image)
        with torch.no_grad():
            text_features = self.encode_text(text)

        return image_features, feature_map, cls_attn


    # def forward(self, image, text):
    #     image_features, feature_map, cls_attn = self.encode_image(image)
    #     with torch.no_grad():
    #         text_features = self.encode_text(text)
    #
    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)
    #
    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     #logits_per_text = logits_per_image.t()
    #
    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict
    '''
    inv_freq = 1. / (10000 ** (torch.arange(0, 2048, 2, dtype=torch.float) / 2048))
    position = torch.arange(50, dtype=torch.float)
    sinusoid_inp = torch.einsum('i,j -> ij', position, inv_freq)
    embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1) / 2048 ** 0.5
    state_dict["visual.attnpool.positional_embedding"] = embeddings
    '''
    #state_dict["visual.positional_embedding"] = upsample_pos_emb(state_dict["visual.positional_embedding"], 28)


    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    # model = CLIP(
    #     embed_dim,
    #     image_resolution, vision_layers, vision_width, vision_patch_size,
    #     context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    # )

    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]

    # CLIP 모델 생성
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )
    
    # ViT-Comer 모듈의 가중치 초기화
    # CNN backbone (SPM)
    model.visual.spm.apply(model.visual._init_weights_fn)
    
    # Interactions (CTI 모듈)
    model.visual.interactions.apply(model.visual._init_weights_fn)
    model.visual.apply(model.visual._init_deform_weights)
    
    # Adapter 모듈
    for adapter in model.visual.adapters:
        adapter.apply(model.visual._init_weights_fn)
    
    # state_dict 로드
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # weights 변환 및 로드
    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)  # strict=False로 설정하여 새로 추가된 모듈을 무시
    
    # ViT parameters freeze
    for name, param in model.visual.transformer.named_parameters():
        param.requires_grad = False
    
    # CNN backbone은 pretrained weights가 없는 경우 학습 가능하게 설정
    # 나머지 모듈(SPM, CTI, Adapter 등)은 learnable하게 설정
    
    return model.eval()

def save_learnable_weights(model, path):
    """Learnable weights (SPM, CTI, Adapter 등) 저장"""
    state_dict = {}
    
    # CNN backbone (SPM) weights
    for name, param in model.visual.spm.named_parameters():
        if param.requires_grad:
            state_dict[f'visual.spm.{name}'] = param
    
    # Interactions (CTI 모듈) weights
    for i, module in enumerate(model.visual.interactions):
        for name, param in module.named_parameters():
            if param.requires_grad:
                state_dict[f'visual.interactions.{i}.{name}'] = param
    
    # Adapter weights
    for i, adapter in enumerate(model.visual.adapters):
        for name, param in adapter.named_parameters():
            state_dict[f'visual.adapters.{i}.{name}'] = param
    
    # 기타 learnable weights
    for name, param in model.visual.named_parameters():
        if param.requires_grad and not any(x in name for x in ['spm', 'interactions', 'adapters', 'transformer']):
            state_dict[f'visual.{name}'] = param
    
    torch.save(state_dict, path)



    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
