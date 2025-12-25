import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x+self.pe[:, :x.size(1)]
    
# class PatchEmbedding(nn.Module):
#     def __init__(self, d_model, patch, stride):
#         super(PatchEmbedding, self).__init__()
#         self.patch_layer = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=patch, stride=stride, padding=patch//2)
#         self.pat
    
#     def forward(self, x):
#         return self.patch_layer(x)

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, time_length, d_model, patch_size, stride=None, pe='zeros', learn_pe=True, dropout=0.1):
        """
        Args:
            time_length: 输入时间序列长度（如24）
            d_model: 输出嵌入维度（如256）
            patch_size: 每个patch的时间步长（如6）
            stride: 滑动步长，默认等于patch_size（无重叠）
        """
        super().__init__()
        self.time_length = time_length
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        # 计算patch数量
        self.num_patches = (time_length - patch_size) // self.stride + 1
        assert self.num_patches > 0, "patch_size或stride过大导致无法分割序列"
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        # # 使用1D卷积同时完成分割和投影
        # self.proj = nn.Conv1d(
        #     in_channels=1,          # 输入通道数（单变量时间序列）
        #     out_channels=d_model,   # 输出嵌入维度
        #     kernel_size=patch_size, # 卷积核大小=patch_size
        #     stride=self.stride      # 步长=stride
        # )
        self.linear= nn.Linear(self.patch_size, d_model)
        # self.norm = nn.LayerNorm(time_length)
        # self.revin_layer = RevIN(time_length, affine=True, subtract_last=False)
        # self.pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model)) # 位置嵌入
        self.week_embedding= nn.Embedding(7, d_model)
        self.interval_embedding= nn.Embedding(96, d_model)

        self.pos_embed = positional_encoding(pe, learn_pe, self.num_patches, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, start_days,start_intervals):
        """
        Args:
            x: 输入张量，形状为 [Batch, num_nodes, time_length]
               （假设num_nodes为节点数，如传感器数量）
        Returns:
            输出张量，形状为 [Batch, num_patches, d_model]
        """
        # batch_size, num_nodes, _ = x.shape
        # x=self.norm(x)
        # x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        x=self.linear(x)
        days=self.week_embedding(start_days)
        intervals=self.interval_embedding(start_intervals)
        x=x+days.unsqueeze(1)+intervals.unsqueeze(1)
        x = self.dropout(x + self.pos_embed)
        return x
        
        # 合并批次和节点维度，适应Conv1d输入要求 [B*N, 1, T]
        # x = x.view(-1, 1, self.time_length)
        
        # # 卷积投影 -> [B*N,  num_patches, d_model]
        # x = self.proj(x).permute(0, 2, 1) 

        # x= self.norm(x)
        
        # # 调整形状 -> [Batch, num_nodes, num_patches, d_model]
        # x = x.view(batch_size, num_nodes, self.num_patches, self.d_model)

        # x = x + self.pos_embed
        
        # return x


# class PatchEmbedding(nn.Module):
#     def __init__(self, time_length, d_model, patch_size, stride=None):
#         """
#         Args:
#             time_length: 输入时间序列的长度
#             d_model: 输出嵌入的维度
#             patch_size: 每个patch的大小，如果为None则默认为time_length
#             stride: patch的步长，如果为None则默认为patch_size（无重叠）
#         """
#         super().__init__()
        
#         self.time_length = time_length
#         self.d_model = d_model
        
#         # 设置默认patch参数
#         if patch_size is None:
#             patch_size = time_length  # 默认不分割，整个序列作为一个patch
#         if stride is None:
#             stride = patch_size  # 默认无重叠
            
#         self.patch_size = patch_size
#         self.stride = stride
        
#         # 计算patch数量
#         self.num_patches = (time_length - patch_size) // stride + 1
        
#         # 投影层：将每个patch投影到d_model维
#         self.proj = nn.Linear(patch_size, d_model)
        
#     def forward(self, x):
#         """
#         Args:
#             x: 输入张量，形状为 (batch_size, num_nodes, time_length)
#         Returns:
#             输出张量，形状为 (batch_size, num_nodes, num_patches, d_model)
#         """
#         batch_size, num_nodes, _ = x.shape
#         # print("before: ",x.dtype)
#         # 分割时间序列为patches
#         # 使用unfold操作高效实现
#         x = x.view(batch_size * num_nodes, 1, 1, -1)  # 调整为适合unfold的4D形状
#         x = F.unfold(x, kernel_size=(1, self.patch_size), stride=(1, self.stride))
        
#         # 调整形状: (batch*num_nodes, patch_size, num_patches) -> (batch, num_nodes, num_patches, patch_size)
#         x = x.permute(0, 2, 1).contiguous()
#         x = x.view(batch_size, num_nodes, self.num_patches, self.patch_size)
        
#         # 投影到d_model维
#         x = self.proj(x)
#         # print("after: ",x.dtype)
#         return x

class CLSToken(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model),requires_grad=True)
        assert self.cls_token.requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            num_instances, seq_len, _ = x.shape
            cls_tokens = self.cls_token.expand(num_instances, 1, -1)
            x = torch.cat([cls_tokens, x], dim=-2)
        elif x.ndim == 4:

            batch_size, num_instances, seq_len, _ = x.shape
            
            # 扩展CLS Token以匹配输入维度
            cls_tokens = self.cls_token.expand(batch_size, num_instances, 1, -1)
            
            # 将CLS Token拼接到每个实例的序列开头
            x = torch.cat([cls_tokens, x], dim=2)  # dim=2对应seq_len维度
        
        return x
    
class LocationEmbedding(nn.Module):
    def __init__(self, d_model, loc_dim=2):
        super(LocationEmbedding, self).__init__()
        # self.pos_table = nn.Parameter(torch.zeros(n_position, d_model))
        self.loc_layer = nn.Linear(loc_dim, d_model)


    def forward(self, loc):
        # Scale lat/lon to range [-1, 1]
        lat_scaled = loc[:,1] * 2 / 90 - 1  # Latitude range [-90, 90]
        lon_scaled = loc[:,0] * 2 / 180 - 1 # Longitude range [-180, 180]
        
        # Stack lat/lon into [N, 2]
        lat_scaled=lat_scaled.unsqueeze(1)
        lon_scaled=lon_scaled.unsqueeze(1)
        loc = torch.cat([lat_scaled, lon_scaled], dim=-1)
        
        # Project to higher dimension
        loc_emb = self.loc_layer(loc)  # [N, dim]
        
        return loc_emb


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()



def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        # pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)




class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x