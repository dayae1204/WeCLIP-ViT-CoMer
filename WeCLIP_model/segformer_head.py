# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .Decoder.TransDecoder import DecoderTransformer

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.proj_2 = nn.Linear(embed_dim, embed_dim)
        # self.proj_3 = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = F.relu(x)
        x = self.proj_2(x)
        return x
    
    
class Conv_Linear(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
        self.proj_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        # self.proj_3 = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x):
        # x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = F.relu(x)
        x = self.proj_2(x)
        return x


# class SegFormerHead(nn.Module):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, in_channels=128, embedding_dim=256, num_classes=20, index=11, **kwargs):
#         super(SegFormerHead, self).__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes

#         self.indexes = index #6 #11

#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

#         linear_layers = [MLP(input_dim=c1_in_channels, embed_dim=embedding_dim) for i in range(self.indexes)]
#         self.linears_modulelist = nn.ModuleList(linear_layers)

#         self.linear_fuse = nn.Conv2d(embedding_dim*self.indexes, embedding_dim, kernel_size=1)
#         self.dropout = nn.Dropout2d(0.1)


#     def forward(self, x_all):
#         x_list = []
#         for ind in range(x_all.shape[0]):
#             x = x_all[ind,:, :, :, :]
#             n, _, h, w = x.shape
#             _x = self.linears_modulelist[ind](x.float()).permute(0,2,1).reshape(n, -1, x.shape[2], x.shape[3])
#             x_list.append(_x)
#         x_list = torch.cat(x_list, dim=1)
#         x = self.linear_fuse(x_list)
#         x = self.dropout(x)

#         return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    - 수정: 단일 텐서(final_cti)도 처리할 수 있도록 변경
    """
    def __init__(self, in_channels=128, embedding_dim=256, num_classes=20, index=11, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.indexes = index #6 #11
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        
        # 기존 코드 유지
        linear_layers = [MLP(input_dim=c1_in_channels, embed_dim=embedding_dim) for i in range(self.indexes)]
        self.linears_modulelist = nn.ModuleList(linear_layers)
        
        self.linear_fuse = nn.Conv2d(embedding_dim*self.indexes, embedding_dim, kernel_size=1)
        
        # 추가: 단일 텐서(final_cti)를 처리하기 위한 1x1 convolution
        self.cti_reducer = nn.Conv2d(768, embedding_dim, kernel_size=1)
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x_all):
        # final_cti가 단일 텐서로 들어왔는지 확인
        if len(x_all.shape) == 4:  # 단일 텐서 [N, C, H, W]
            # final_cti 직접 처리
            x = self.cti_reducer(x_all)
            x = self.dropout(x)
            return x
        
        # 기존 처리 방식 (여러 레이어의 특징을 처리)
        x_list = []
        for ind in range(x_all.shape[0]):
            x = x_all[ind,:, :, :, :]
            n, _, h, w = x.shape
            _x = self.linears_modulelist[ind](x.float()).permute(0,2,1).reshape(n, -1, x.shape[2], x.shape[3])
            x_list.append(_x)
        x_list = torch.cat(x_list, dim=1)
        x = self.linear_fuse(x_list)
        x = self.dropout(x)
        
        return x