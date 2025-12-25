import math
import torch
import torch.nn as nn
from torch.nn import init
import time
import random
import torch.nn.functional as F
from models.layers.Embedding import *
from torch.amp import autocast
from flash_attn import flash_attn_func 
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def chunked_flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            dropout_p:float = 0.0, causal: bool = False, bs_max: int = 50000):
    B = q.shape[0]
    outs = []
    for i in range(0, B, bs_max):
        q_chunk = q[i:i+bs_max]
        k_chunk = k[i:i+bs_max]
        v_chunk = v[i:i+bs_max]
        out_chunk = flash_attn_func(q_chunk, k_chunk, v_chunk, 
                                    dropout_p=dropout_p,
                                    causal=causal)
        outs.append(out_chunk)
    return torch.cat(outs, dim=0)


class MultimodalAttention(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_layers: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 num_modality=4,
                 alpha=0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modal_num = num_modality
        self.alpha = alpha
        self.W = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.norm1= nn.LayerNorm(self.hidden_dim) 
        self.dropout1= nn.Dropout(dropout)
        
    def forward(self, multimodal: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        if adj is not None:
            N, M, D = multimodal.shape
            out1 = torch.sparse.mm(adj, multimodal.reshape(N, -1)) 
            out1 = torch.mm(out1.reshape(N*M,-1), self.W).reshape(N,M,D)  
            out =self.norm1(multimodal + self.alpha*self.dropout1(out1))
        return out

def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device, factor=1.0):
    position = torch.arange(0, max_len * factor, 1 / factor, dtype=torch.float).unsqueeze(-1)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  
    theta = torch.pow(10000, -2 * ids / output_dim)
    embeddings = position * theta
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, -1, output_dim))
    embeddings = embeddings.to(device)

    if factor > 1.0:
        interpolation_indices = torch.linspace(0, embeddings.shape[2] - 1, max_len).long()
        embeddings = embeddings[:, :, interpolation_indices, :]
    return embeddings

def RoPE(q, k):
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device, factor=1)

    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  

    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape) 

    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos

    return q, k


def RoPE_decoder(q, k):
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    q_max_len = q.shape[2]
    k_max_len = k.shape[2]
    output_dim = q.shape[-1]

    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, k_max_len + q_max_len, output_dim, q.device, factor=1)

    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  

    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  

    q = q * cos_pos[:,:,-q_max_len:,:] + q2 * sin_pos[:,:,-q_max_len:,:]


    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos[:,:,:k_max_len,:] + k2 * sin_pos[:,:,:k_max_len,:]
    return q, k


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False, rope_type=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.rope_type = rope_type

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, prev:Optional[torch.Tensor]=None, key_padding_mask:Optional[torch.Tensor]=None, attn_mask:Optional[torch.Tensor]=None):
        # using RoPE
        if self.rope_type:
            q, k = RoPE_decoder(q, k.permute(0,1,3,2).contiguous())
        else:
            q, k = RoPE(q, k.permute(0,1,3,2).contiguous())
        k = k.permute(0,1,3,2).contiguous()

        attn_scores = torch.matmul(q, k) * self.scale    

        if prev is not None: attn_scores = attn_scores + prev

        if attn_mask is not None:                                   
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:                          
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)              
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)                   

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, rope_type=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, self.d_k * num_heads)
        self.W_K = nn.Linear(d_model, self.d_k * num_heads)
        self.W_V = nn.Linear(d_model, self.d_k * num_heads)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, num_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa, rope_type=rope_type)

        self.to_out = nn.Sequential(nn.Linear(num_heads * self.d_k, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q, K, V, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)

        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q)
        q_s=q_s.reshape(bs, -1, self.num_heads, self.d_k).permute(0,2,1,3).contiguous() 
        k_s = self.W_K(K).reshape(bs, -1, self.num_heads, self.d_k).permute(0,2,3,1).contiguous()     
        v_s = self.W_V(V).reshape(bs, -1, self.num_heads, self.d_k).transpose(1,2).contiguous()      


        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().reshape(bs, -1, self.num_heads * self.d_k)
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(gelu(self.linear1(x))))
    
def gelu(x):
    """Implementation of the GELU activation function.
    For information: OpenAI GPT's GELU is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.res_attention = res_attention

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_model, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_model, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.pre_norm = False
        self.store_attn = False

    def forward(self, x, prev=None, key_padding_mask=None, attn_mask=None):
        if self.res_attention:
            src2, attn, scores = self.attention(x, x, x, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.attention(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.dropout_attn(src2)

        src = self.norm_attn(x)

        src2 = self.ff(src)

        ## Add & Norm
        src = src + self.dropout_ffn(src2) 

        src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

    
class SparseSelector(nn.Module):
    def __init__(self, d_model, num_heads, num_modals =5):
        super().__init__()
        self.selector = LearnableSparseSelector(d_model, num_heads, num_modals=num_modals)
        self.ffn = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, state_expanded, x_emb=None, mask=None):
        select_output, x_emb, binarymask = self.selector(x, state_expanded, x_emb)
        x=select_output
        if x_emb is not None:
            return x, x_emb, binarymask
        return x, binarymask

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model=512, num_heads=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.cls_token = CLSToken(d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, num_sensors, seq_len, d_model = x.shape
        x = self.cls_token(x)
        x  = x.reshape(batch*num_sensors, seq_len+1, d_model)
        for layer in self.layers:
            x = layer(x)
        x  = x.reshape(batch, num_sensors, -1, d_model)
        return self.linear(x)  

       



def generate_tgt_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)



class BinaryMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_probs):
        max_indices = torch.argmax(gate_probs, dim=1, keepdim=True)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, max_indices, 1)
        mask = torch.logical_or(mask, (gate_probs > 0.8)).float()
        
        return mask
   
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 

     
class LearnableSparseSelector(nn.Module):   
    def __init__(self, d_model, num_heads, hidden_dim=128,num_modals =5,training=True,tau=0.5,     
                 eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_modals = num_modals   
        self.tau = tau
        self.eps = eps
        self.training=training
        self.uncertainty_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Dropout(0.1),
                nn.Sigmoid() 
            ) for dim in range(self.num_modals)
        ])

        
        self.importance_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*d_model, hidden_dim),
                nn.Dropout(0.1),
                nn.Sigmoid()
                ) for dim in range(self.num_modals)
        ])

        self.q_proj = nn.Linear(d_model, d_model*num_heads)
        self.k_proj = nn.Linear(d_model, d_model*num_heads)
        self.v_proj = nn.Linear(d_model, d_model*num_heads)
    
    def gumbel_sigmoid(self, logits, temp=None):
        """可微分二值化门控 """
        alpha=0.05
        temp = self.tau if temp is None else temp
        noise1 = torch.rand_like(logits)*alpha
        noise2 = torch.rand_like(logits)*alpha
        gumbel_noise1 = -torch.log(-torch.log(noise1 + self.eps) + self.eps) 
        gumbel_noise2 = -torch.log(-torch.log(noise2 + self.eps) + self.eps)
        if self.training: 
            y = (logits + gumbel_noise1 - gumbel_noise2) / temp
        else: 
            y = (logits) / temp
        return torch.sigmoid(y)  
    
    def gumbel_softmax_binary(self, logits, temp=None, num_samples=5):
        """
        实现图中的Gumbel Softmax二值化
        logits: 原始分数 [*, M, M]
        tau: 温度参数
        num_samples: 蒙特卡洛采样数
        """
        temp = self.tau if temp is None else temp
        g1 = -torch.log(-torch.log(torch.rand(num_samples, *logits.shape)))
        g0 = -torch.log(-torch.log(torch.rand(num_samples, *logits.shape)))
        
        s_relaxed = torch.sigmoid((torch.logit(torch.sigmoid(logits)) + g1 - g0) / temp)

        return s_relaxed.mean(dim=0) 

        
    def forward(self, q,state_expanded, x_emb=None, mask=None):
        N, M, D = q.shape
        if x_emb is not None:
            B, N , L, D =x_emb.shape
            x_emb=self.v_proj(x_emb).reshape(B, N, L, self.num_heads, D).mean(-2)
        query=self.q_proj(q).reshape(N, M, self.num_heads, D).transpose(1, 2)
        value=self.v_proj(q).reshape(N, M, self.num_heads, D).transpose(1, 2)
        modal_features = []
        last_modal_feat = query[:, :, -1, :]
        modal_importances = []
        for i in range(self.num_modals):
            modal_feat = query[:, :, i, :]  
            modal_features.append(modal_feat)
            combined_feat = torch.cat([modal_feat, last_modal_feat], dim=-1)
            proj  = self.importance_projectors[i](combined_feat) 
            importance = proj.mean(dim=-1, keepdim=True)
            modal_importances.append(importance)
        importance_scores  = torch.stack(modal_importances, dim=2)  

        uncertainties = []
        for i, (feat, predictor) in enumerate(zip(modal_features, self.uncertainty_predictors)):
            u=predictor(feat)
            uncertainties.append(u)
        uncertainty_scores = torch.stack(uncertainties, dim=2) 

        combined_logits = importance_scores / (self.eps + uncertainty_scores)  
        gate_probs = self.gumbel_sigmoid(combined_logits)  
        full_gate_mask=gate_probs.mean(dim=1)*state_expanded
        binary_mask = BinaryMaskSTE.apply(full_gate_mask)

        output = value.mean(dim=1)  
        return output, x_emb, binary_mask


class MultiModalSelector(nn.Module):
    def __init__(self, input_dim, d_model=512, num_heads=1, num_layers=6,num_modal=4,args=None):  
        super().__init__()
        self.poi_layer = nn.Linear(args.model.poi_dim,input_dim)
        self.poi_embedding= nn.Parameter(torch.randn(input_dim))
        self.satellite_layer = nn.Linear(args.model.satellite_dim,input_dim)
        self.satellite_embedding= nn.Parameter(torch.randn(input_dim))
        self.x_layer = nn.Linear(input_dim,input_dim)
        self.x_embedding= nn.Parameter(torch.randn(input_dim))
        self.loc_layer = LocationEmbedding(input_dim,args.model.loc_dim )
        self.loc_embedding= nn.Parameter(torch.randn(input_dim))
        self.pos=PositionalEmbedding(input_dim)        
        self.layer = SparseSelector(input_dim, num_heads,num_modals=num_modal)
        self.linear = nn.Linear(input_dim, d_model,bias=False)
        self.args=args

    
    def random_zero_per_row_advanced(self, state):
        state1 = state.squeeze(-1)  
        ones_per_row = state1.sum(dim=1)
        
        prob_dist = state1.float() / state1.sum(dim=1, keepdim=True).clamp(min=1)
        selected_indices = torch.full((state.size(0),), -1, dtype=torch.long, device=state.device)
        mask = torch.zeros_like(state, dtype=torch.bool)
        rows_to_mask = (ones_per_row > 1)
        
        if rows_to_mask.any():
            with torch.no_grad():
                selected_indices[rows_to_mask] = torch.multinomial(
                    prob_dist[rows_to_mask], 
                    num_samples=1
                ).squeeze(-1)
            
            mask[rows_to_mask, selected_indices[rows_to_mask]] = True
        state_neg = torch.where(
            mask,
            torch.zeros_like(state),  
            state                     
        )
        return state_neg, selected_indices
    
    def replace_with_noise(self, modality_embeddings, selected_indices, noise_scale=1):
        
        N, M, D = modality_embeddings.shape
        device = modality_embeddings.device
    
        
        noise = torch.randn(N, M, D, device=device) * noise_scale
        

        mask = torch.zeros(N, M, D, device=device, dtype=torch.bool)
        valid_rows = selected_indices != -1
        valid_rows = valid_rows.to(device)
        if valid_rows.any(): 
            mask[valid_rows, selected_indices[valid_rows], :] = True
        
        modified_embeddings = torch.where(
            mask,
            noise,
            modality_embeddings
        )
        
        return modified_embeddings

    def forward(self, x, poi, satellite, loc, state):
        B,N,L,D=x.shape
        poi_emb = self.poi_layer(poi)+self.poi_embedding   
        satellite_emb = self.satellite_layer(satellite)+self.satellite_embedding
        x_emb = self.x_layer(x)+self.x_embedding
        loc_emb = self.loc_layer(loc)+self.loc_embedding
        state = state.reshape(poi_emb.shape[0], -1)  
        x_cls_emb=x_emb[:,:,0,:].mean(dim=0)
        modality_matrix = torch.stack([poi_emb, satellite_emb, loc_emb, x_cls_emb], dim=1)  
        state_expanded = state.unsqueeze(-1)  
        emb = modality_matrix    
        emb1,x_emb,binary_mask = self.layer(emb,state_expanded,x_emb)
        emb1= self.linear(emb1)
        x_emb= self.linear(x_emb)
        if self.args.is_training:
            state_neg, selected_indices=self.random_zero_per_row_advanced(binary_mask)
            emb3=self.replace_with_noise(emb,selected_indices)
        # 输出层
            emb2,binary_mask1=emb1,binary_mask
            binary_mask1=binary_mask1* state_neg
            emb2 = self.linear(emb2)
            emb3,binary_mask2=self.layer(emb3,state_expanded)
            emb3 = self.linear(emb3)
        else:
            emb2,emb3=None,None
            binary_mask1,binary_mask2=None,None
            return emb1,emb2,emb3,x_emb,binary_mask* state_expanded,binary_mask1,binary_mask2 
        return emb1,emb2,emb3,x_emb,binary_mask,binary_mask1,binary_mask2 * state_expanded
    
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x 
    
class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(ks,c_in, ks,c_out))  
        self.b = nn.Parameter(torch.FloatTensor(1, 1, ks, c_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, adj):
        B, N, L, D = x.shape
        x=x.reshape(B,N,L*D)
        x_c = torch.stack([torch.sparse.mm(adj, x[i]) for i in range(B)]).reshape(B, N, L , D)
        x_gc = torch.einsum('bnld,ldjk->bnjk', x_c, self.theta)  + self.b 
        return torch.relu(x_gc + x.reshape(B,N,L,D)) 
    
class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        padding_height = (kt - 1) // 2 if kt % 2 == 1 else kt // 2
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1, padding=[padding_height, 0])
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1, padding=[int((kt-1)/2), 0])

    def forward(self, x):
        x_in = self.align(x)
        if self.act == "GLU":
            x_conv = self.conv(x)
            
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in) 
        return torch.relu(self.conv(x) + x_in) 
    
class FullyConvLayer(nn.Module):
    def __init__(self, c, out_dim):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1) 

    def forward(self, x):
        return self.conv(x)
    

class ManualLayerNorm(nn.Module):
    def __init__(self, d_model,normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape  
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        nn.init.uniform_(self.gamma, a=-math.sqrt(5), b=math.sqrt(5))

    def forward(self, x):
        mean = x.mean(dim=self.normalized_shape, keepdim=True)  
        var = x.var(dim=self.normalized_shape, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm

    
class STConv(nn.Module):
    def __init__(self, kt, d_model, d_ff, all_token_num, token_num, dropout=0.05):
        super().__init__()
        self.token_num=token_num
        self.all_token_num=all_token_num
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.tconv1 = TemporalConvLayer(kt, d_model, d_ff, "GLU")
        self.sconv1 = SpatioConvLayer(all_token_num, d_ff, d_ff)
        self.tconv2 = TemporalConvLayer(kt, d_ff,d_model)
        self.sconv2 = SpatioConvLayer(all_token_num, d_model, d_model)
        self.manualnorm1 =ManualLayerNorm(d_model,(-2,-1))
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, adj):
        x_in=x[:,:,-self.token_num:,:]   
        x_mm=x[:,:,:-self.token_num,:]
        x_mm=self.linear1(x_mm)
        x_in=x_in.permute(0,3,2,1).contiguous()  
        x_in=self.tconv1(x_in).permute(0,3,2,1).contiguous()  
        x=torch.cat([x_mm, x_in], dim=-2)
        x= self.sconv1(x, adj)
        x_in=x[:,:,-self.token_num:,:]   
        x_mm=x[:,:,:-self.token_num,:]
        x_mm=self.linear2(x_mm)
        x_in=self.tconv2(x_in.permute(0,3,2,1).contiguous())
        x_in=x_in.permute(0,3,2,1).contiguous()
        x=torch.cat([x_mm, x_in], dim=-2) #B N L D
        x=self.manualnorm1(x).permute(0,3,1,2).contiguous()   
        return self.dropout1(x)


def causal_attention_mask(seq_length,num):
    mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(num,1,seq_length, seq_length)

class PredictHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = self.linear( self.dropout(x) )    
        return x

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0,2,1)
        return x

class LastDecoderLayer(nn.Module):
    def __init__(self, patch_len, d_model, n_heads, d_ff=None, attn_dropout = 0.2, dropout=0.5, norm="BatchNorm"):
        super(LastDecoderLayer, self).__init__()
        self.n_heads=n_heads
        self.self_attn = nn.Linear(d_model, d_model * 3 )  
        self.cross_attn = nn.Linear(d_model, d_model)      
        self.kv_proj = nn.Linear(d_model, d_model * 2)   
        self.out_proj1= nn.Linear(d_model, d_model)
        self.out_proj2= nn.Linear(d_model, d_model)
        self.p=dropout
        
        if 'batch' in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)


        self.dropout = nn.Dropout(dropout)

        self.MLP1 = CMlp(in_features = d_model, hidden_features = d_ff, out_features = d_model, drop=dropout)

    def forward(self, x, cross):
        batch, num_patch, d_model = x.shape
        batch, num_tokens, d_model = cross.shape
        qkv = self.self_attn(x)
        qkv=qkv.reshape(batch, num_patch, self.n_heads, -1)
        q, k, v = qkv.chunk(3, dim=-1)  

        x_attn = chunked_flash_attn_func(
                q, k, v,
                dropout_p=self.p if self.training else 0.0,
                causal=True
            ).reshape(batch, num_patch, -1)
        x_attn = self.out_proj1(x_attn)
        x_attn = self.norm1(x_attn) + x
        q_cross = self.cross_attn(x_attn).reshape(batch, num_patch, self.n_heads, -1)
        kv = self.kv_proj(cross). reshape(batch, num_tokens, self.n_heads, -1)
        k,v=kv.chunk(2, dim=-1)
        x_cross = chunked_flash_attn_func(
                q_cross, k, v,
                dropout_p=self.p if self.training else 0.0,
                causal=False
            ).reshape(batch, num_patch, -1)
        x_cross = self.out_proj2(x_cross)
        x_cross = self.dropout(self.norm2(x_cross)) + x_attn
        x_ff = self.MLP1(x_cross)
        x_ff = self.norm3(x_ff) + x_cross
        return x_ff
    



class CrossNode(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout = 0.1, dropout=0.6, norm="BatchNorm",use_weights=True,k=20):
        super().__init__()
        self.use_weights = use_weights
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.k=k   
        self.p=dropout
        
        if 'batch' in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, neighbor_feats, topk_values):
        N, B,K, D= neighbor_feats.shape
        N, B, _,D =x.shape    
        x=x.reshape(-1,_,D)    
        topk_values=topk_values.reshape(-1,1,K,1)
        if self.use_weights:
            neighbor_feats = neighbor_feats * topk_values.expand(-1, 1, K, D)  
        neighbor_feats=neighbor_feats.reshape(-1,K,D)
        q_x= self.q_proj(x).reshape(B*N,1,self.n_heads,-1).contiguous()
        k_neighbor_feats= self.k_proj(neighbor_feats).reshape(B*N,K,self.n_heads,-1).contiguous()
        v_neighbor_feats= self.v_proj(neighbor_feats).reshape(B*N,K,self.n_heads,-1).contiguous()
        x_cross=chunked_flash_attn_func(
                q_x, k_neighbor_feats, v_neighbor_feats,
                dropout_p=self.p if self.training else 0.0,
                causal=False
            ).reshape(-1, 1, D)
        x_cross=self.out_proj(x_cross)
        x_cross = self.dropout(self.norm1(x_cross))+ x

        return x_cross

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        
        super().__init__()
        self.top_k = top_k
        self.fc = nn.Linear(input_dim, num_experts)
        
        
    def forward(self, x):
        logits = self.fc(x)  
        return F.softmax(logits, dim=-1)


class mMOE(nn.Module):
    def __init__(self,d_model, n_heads, num_experts, modality_num=4, attn_dropout=0.05,top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_modality = modality_num
        self.router = Router(d_model, num_experts, top_k)
        
        self.modality_experts= nn.ModuleList([CrossNode(d_model, n_heads, attn_dropout=attn_dropout, dropout=0.3)for _ in range(modality_num)])  # 0.6
        self.experts= nn.ModuleList([CrossNode(d_model, n_heads, attn_dropout=attn_dropout, dropout=0.3)for _ in range(num_experts)])   # 0.7
        self.linear1= nn.ModuleList([nn.Linear(d_model, d_model)for _ in range(modality_num)])  
        self.linear2= nn.ModuleList([nn.Linear(d_model, d_model)for _ in range(num_experts)]) 

    def forward(self, x, mm, topk_indices, topk_values, modality_index):
        N, B, L, D = x.shape
        N,K=topk_indices.shape
        expanded_indices = topk_indices.reshape(N, K, 1, 1, 1).expand(-1, -1, B, L, D)
        neighbor_feats = torch.gather(
            x.unsqueeze(1).expand(-1, K, -1, -1, -1), 
            dim=0, 
            index=expanded_indices
        ).permute(0,2,3,1,4).contiguous()  
        neighbor_feats=neighbor_feats.reshape(N,-1,K,D)
        x=x.reshape(N,-1,1,D)
        topk_values=topk_values.reshape(N,K,1)
        gate_weights = self.router(mm)  #[N E]
        topk_weights, topk_experts = gate_weights.topk(self.top_k, dim=-1)  #N 2

        routed_experts = torch.zeros_like(gate_weights).scatter_(
            dim=-1,
            index=topk_experts,
            src=torch.ones_like(topk_weights),
        )

        balance_loss = 0
        total_tokens = x.shape[0]     
        f_i = torch.sum(routed_experts, dim=(0)) * (1 / total_tokens)
        P_i = torch.sum(gate_weights, dim=(0)) * (1 / total_tokens)
        balance_loss = self.num_experts *torch.sum((f_i*P_i))

        flat_routed_experts = routed_experts.view(-1, self.num_experts)
        total_expert_allocation = torch.cumsum(flat_routed_experts, dim=0)
        expert_capacity = N / self.num_experts
        expert_mask = (total_expert_allocation <= expert_capacity).float()
        revised_expert_allocation = expert_mask * flat_routed_experts
        routed_experts = revised_expert_allocation.view(
            routed_experts.shape
        ) 
        routed_expert_probs = gate_weights * routed_experts

        active_tokens = (routed_expert_probs.sum(dim=-1) > 0).view(-1)
        expert_probs,expert_indices = routed_expert_probs.topk(self.top_k, dim=-1)
        active_experts = expert_indices[active_tokens]
        active_x = x[active_tokens]  # N B*L D
        active_neighbor_feats = neighbor_feats[active_tokens]  # N B*L K D
        active_topk_values=topk_values[active_tokens]
        active_out = torch.zeros_like(active_x)  # N B*L D
        expert_probs=expert_probs.to(active_out.dtype)
        for i in range(self.num_experts):
            mask=(active_experts ==i).any(dim=2).squeeze(-1)
            if mask.any():
                M,_,_,_=active_x[mask].shape
                expert_out= self.experts[i](active_x[mask], active_neighbor_feats[mask], active_topk_values[mask]).reshape(-1,B*L,1,D)
                expert_out=self.linear1[i](expert_out)  
                active_out[mask] = active_out[mask]+ expert_out*expert_probs[active_tokens][active_experts==i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = torch.zeros_like(x)
        out[active_tokens] = active_out
        
        
        modality_index=modality_index.to(expert_indices.dtype)
        weights = modality_index / modality_index.sum(dim=1, keepdim=True).to(active_out.dtype)  
        active_out = torch.zeros_like(x) 
        for i in range(self.num_modality):
            mask = (modality_index[:, i, :] == 1).any(dim=1)
            if mask.any():
                M,_,_,_=x[mask].shape
                expert_out= self.modality_experts[i](x[mask],neighbor_feats[mask], topk_values[mask]).reshape(-1,B*L,1,D)
                expert_out=self.linear2[i](expert_out)  
                active_out[mask] += expert_out 
        
        out=(out+1/self.num_modality*active_out)
        out = out.reshape(N, B, L, D)
        
        return out, balance_loss
    
def create_value_mask(topk_values, num, head, min_threshold=0.01):
    N,_, K,_= topk_values.shape
    topk_values = topk_values.squeeze(-1)  
    bool_mask = topk_values < min_threshold
    
    float_mask = torch.where(
        bool_mask,
        True,  
        False            
    ).unsqueeze(1).unsqueeze(0).expand(num,N,head, 1,K).reshape(-1,head,1,K)
    
    return float_mask.to(topk_values.device)  

def create_attention_mask(binary):
    binary = binary.permute(0,2,1)
    N,_,M = binary.shape
    mask = binary.unsqueeze(1)
    mask = (mask == 0).float().masked_fill(mask == 0, float('-inf'))
    
    return mask  
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, num_experts=4, attn_dropout=0.1,top_k=1, dropout=0.5, modality_num=4,norm="BatchNorm",use_weights=True):
        super(DecoderLayer, self).__init__()
        self.use_weights = use_weights
        self.moe=mMOE(d_model, n_heads, num_experts=num_experts, modality_num=modality_num,attn_dropout=0.05,top_k=top_k)
        self.dropout = nn.Dropout(dropout)



    def forward(self, mm, x, topk_indices, topk_values, modality_index):
        N, K= topk_values.shape
        B, N, L, D= x.shape
        x=x.permute(1,0,2,3).contiguous()
        x, balance_loss =self.moe(x, mm, topk_indices, topk_values, modality_index)
        x=x.reshape(B,N,L,D)
        return x, balance_loss
        

class Decoder(nn.Module):
    def __init__(self, d_model, pred_len, seq_len, num_heads, patch_len,d_layers,d_ff=128, dropout=0.1,attn_dropout=0.1, token_num=6, modality_num=4,head_dropout=0.1,kt1=5,kt2=3,use_weights=True, num_experts=4, topk=1,last_layer=3):
        super().__init__()
        self.decoder_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decoder_layers.append(DecoderLayer(d_model, num_heads, d_ff,  num_experts, attn_dropout,topk, dropout,modality_num,use_weights=use_weights))
        self.dropout1 = nn.Dropout(dropout)
        self.pred_len = pred_len
        self.out_patch_num = math.ceil(pred_len / patch_len)
        self.seq_len = seq_len
        self.d_model = d_model
        self.token_num = token_num
        self.all_token_num= token_num+1+modality_num
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout2 = nn.Dropout(dropout)
        self.hidden_dim=d_ff
        self.use_weights=use_weights
        self.pre_linear=PredictHead(d_model, patch_len, head_dropout)
        self.lastdecoder_layers = nn.ModuleList()
        for i in range(last_layer):
            self.lastdecoder_layers.append(LastDecoderLayer(patch_len, d_model, num_heads, d_ff, attn_dropout, dropout))

    def get_dynamic_weights(self, n_preds, decay_rate=0.5):
        weights = decay_rate ** torch.arange(n_preds)
        return weights

    def forward(self, mm, x, topk_indices, topk_values, modality_index):   
        B,N,L, D = x.shape
        all_banlance_loss=[]
        
        N, K= topk_indices.shape
        cross_x = x[:,:,-1:,:]
        for layer in self.decoder_layers:
            cross_x, banlance_loss = layer(mm, cross_x, topk_indices, topk_values, modality_index)
            all_banlance_loss.append(banlance_loss)
        if all_banlance_loss[0]:
            all_banlance_loss=torch.sum(torch.stack(all_banlance_loss,dim=0)) 
        else:
            all_banlance_loss=0
        cross_x=torch.cat((x[:,:,:-1,:],cross_x),dim=2)
        x=x.reshape(-1,L,D)
        cross_x=cross_x.reshape(-1,L,D)
        x_last=x[:,-1,:].unsqueeze(1).expand(-1,self.out_patch_num,-1)
        weights = self.get_dynamic_weights(self.out_patch_num).to(x_last.device)
        x_last = x_last * weights.unsqueeze(0).unsqueeze(-1)
        for layer in self.lastdecoder_layers:
            x_last=layer(x_last,cross_x)
        x=x_last.reshape(B,N,-1,D)
        x=self.pre_linear(x)
        x=self.flatten(x)
        return x[:,:,:self.pred_len], all_banlance_loss