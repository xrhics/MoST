from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import ConfigDict, load_config
from models.layers.Layer import Encoder, Decoder,MultiModalSelector,FullyConvLayer,MultimodalAttention,ManualLayerNorm,create_attention_mask,MultiHeadAttention
from models.layers.Embedding import PatchEmbedding,CLSToken
import gc
from collections import defaultdict
class Model(nn.Module):
    def __init__(self, dropout=0.1, init_temp=1.0,init_margin=5.0,args: ConfigDict = None):  
        super().__init__()
        self.args=args
        device=torch.device('cuda:{}'.format(args.gpu))
        self.vocab_size = args.model.MM.vocab_size
        self.patch = args.model.MM.patch   
        self.stride = args.model.MM.stride  
        self.input_len=args.model.seq_len
        self.pred_len=args.model.pred_len
        pred_len=self.pred_len
        self.input_dim=args.model.encoder.dim
        d_model=self.input_dim
        self.num_heads=args.model.encoder.num_heads
        self.num_modality=args.data.num_modality
        self.hidden_dim=args.model.MM.hidden_dim
        d_ff=self.hidden_dim
        self.encoder_num_layers=args.model.encoder.num_layers
        self.decoder_num_layers=args.model.decoder.num_layers
        self.mm_num_layers=args.model.mm_num_layers
        self.num_experts=args.model.num_experts
        self.topk=args.model.top_k
        self.margin =init_margin
        self.patchEmbedding=PatchEmbedding(self.input_len, d_model,  self.patch, self.stride)
        self.encoder = Encoder(input_dim=d_model, d_model=d_model, num_heads=self.num_heads, num_layers=self.encoder_num_layers)
        self.decoder = Decoder(d_model=d_ff, pred_len=pred_len, seq_len=self.input_len, num_heads=self.num_heads, token_num=(self.input_len - self.patch) // self.stride + 2,patch_len=self.patch,d_layers=self.decoder_num_layers,d_ff=d_ff, num_experts=self.num_experts, topk=self.topk, modality_num=self.num_modality) 
        self.multimodel_selector=MultiModalSelector(d_model, self.hidden_dim, args=args)
        self.norm= nn.LayerNorm(self.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model),requires_grad=True)
        self.mm_attn=MultiHeadAttention(d_model, 4, attn_dropout=0.1)


    def forward(self, x, start_days,start_intervals, poi, satellite, loc, adj, input_mean,input_std, state, topk_indices, topk_values):
        x=self.patchEmbedding(x,start_days,start_intervals) 
        value=self.encoder(x)   
        B,N,L,D=value.shape
        multimodal, multimodal_del, multimodal_noise,x_emb,binary_mask,binary_mask1,binary_mask2=self.multimodel_selector(value, poi, satellite, loc, state)  ##multi-model  filter  x, poi, satellite, map, loc, state
        multimodal=self.norm(multimodal)
        modality_index=binary_mask
        
        binary_mask=create_attention_mask(binary_mask)
        cls_token=self.cls_token.expand(N, 1, -1)
        mm,_=self.mm_attn(cls_token,multimodal,multimodal,attn_mask=binary_mask)
        x1, balance_loss1=None,None
        if self.training:
            multimodal_del=self.norm(multimodal_del)
            multimodal_noise=self.norm(multimodal_noise)
            binary_mask1=create_attention_mask(binary_mask1)
            binary_mask2=create_attention_mask(binary_mask2)
            multimodal_del,_=self.mm_attn(cls_token,multimodal_del,multimodal_del,attn_mask=binary_mask1)
            multimodal_noise,_=self.mm_attn(cls_token,multimodal_noise,multimodal_noise,attn_mask=binary_mask2)
        x, balance_loss = self.decoder(mm,x_emb[:,:,1:,:], topk_indices, topk_values,modality_index)
        x=x*input_std+input_mean
        return x, mm, multimodal_del, multimodal_noise, balance_loss
    def get_balance_loss(self, gate_weights,all_topK_experts):
        expert_counts = torch.zeros(self.num_experts, device=gate_weights.device)
        counts = torch.bincount(all_topK_experts.flatten(), minlength=self.num_experts)
        expert_counts += counts.float()
        f_i = expert_counts / expert_counts.sum() 
        P_i = gate_weights.mean(dim=[0, 1]) 
        alpha=0.1
        loss = alpha * self.num_experts * torch.sum(f_i * P_i)
        return loss
    

    def get_nearest_embedding(self, idxs):
        return self.quantizer.codebook(idxs)

    def get_next_autoregressive_input(self, idx, f_hat_BCHW, h_BChw):
        return self.quantizer.get_next_autoregressive_input(idx, f_hat_BCHW, h_BChw)

    def to_img(self, f_hat_BCHW):
        return self.decoder(f_hat_BCHW).clamp(-1, 1)

    def img_to_indices(self, x):
        f = self.encoder(x)
        fhat, r_maps, idxs, scales, loss = self.quantizer(f)
        return idxs

    def get_loss(self, x, x_hat):
        recon_loss = F.mse_loss(x_hat,x) 
        return recon_loss
    
    def get_contrastive_loss(self, multimodal1, multimodal2, multimodal3):
        # multimodal1, multimodal2, multimodal3: [N, D]
        multimodal1 = multimodal1.reshape(multimodal1.shape[0], -1)
        multimodal2 = multimodal2.reshape(multimodal2.shape[0], -1)
        multimodal3 = multimodal3.reshape(multimodal3.shape[0], -1)
        neg_dist1 = F.pairwise_distance(multimodal1, multimodal2, p=2)  # [N]
        neg_dist2 = F.pairwise_distance(multimodal1, multimodal3, p=2)  # [N]
        negs_dist = F.pairwise_distance(multimodal2, multimodal3, p=2) # [N]

        margin2 = 1 

        neg_loss1 = 0.5 * torch.pow(torch.clamp(self.margin - neg_dist1, min=0.0), 2).mean()
        neg_loss2 = 0.5 * torch.pow(torch.clamp(self.margin - neg_dist2, min=0.0), 2).mean()
        neg_sim_loss = negs_dist.mean() * margin2

        loss = neg_loss1+ neg_loss2 + neg_sim_loss
        return loss


    def get_setting(self, args):
        setting = 'MM_lightning_lr{}_bs{}_sl{}_pl{}_edim{}_vs{}_patch{}'.format(
            args.train.lr,
            args.data.batch_size,
            args.model.seq_len,
            args.model.pred_len,
            args.model.encoder.dim,
            args.model.MM.vocab_size,
            args.model.MM.patch
        )
        return setting