import torch
import torch.nn as nn
import math
from .base import ModelBase

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        self.init_encoding()

    def init_encoding(self):
        position = torch.arange(0, self.pe.size(0), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pe.size(1), 2).float() * (-math.log(10000.0) / self.pe.size(1)))
        pe = torch.zeros_like(self.pe)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe.data.copy_(pe)  # Properly initialize the learnable parameter

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = query

        qkv = self.qkv(query).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Sparse Attention Mechanism
        attn = self.prob_sparse_attention(q, k)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.fc(out)
        return out

    def prob_sparse_attention(self, q, k):
        B, H, N, D = q.shape
        U = min(self.factor * math.ceil(math.log(N)), N)

        # Compute importance scores
        query_norms = torch.norm(q, dim=-1)  # (B, H, N)
        index = torch.topk(query_norms, U, dim=-1)[1]  # Get indices of top-U queries

        # Gather selected queries
        q_reduced = torch.gather(q, 2, index.unsqueeze(-1).expand(-1, -1, -1, D))

        # Compute sparse attention
        attn = (q_reduced @ k.transpose(-2, -1)) * (D ** -0.5)
        attn = attn.softmax(dim=-1)

        # Expand back to full shape
        full_attn = torch.zeros(B, H, N, N, device=q.device)
        full_attn.scatter_(2, index.unsqueeze(-1).expand(-1, -1, -1, N), attn)

        # Mask out irrelevant attention scores
        mask = torch.zeros_like(full_attn, dtype=torch.bool)
        mask.scatter_(2, index.unsqueeze(-1).expand(-1, -1, -1, N), True)
        full_attn = full_attn.masked_fill(~mask, float('-inf'))
        full_attn = full_attn.softmax(dim=-1)

        return full_attn

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(InformerEncoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(InformerDecoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        self_attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Informer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, d_ff=512, num_encoder_layers=2, num_decoder_layers=1, dropout=0.1):
        super(Informer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = LearnablePositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([InformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.memory_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        memory = self.memory_proj(x)
        for layer in self.decoder_layers:
            x = layer(x, memory)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x

class InformerModel(ModelBase):
    def __init__(self, input_dim, savedir, **kwargs):
        model = Informer(input_dim=input_dim)
        model_weight = "output_proj.weight"
        model_bias = "output_proj.bias"
        model_type = "informer"
        self.device = kwargs.get('device', torch.device("cpu"))
        super(InformerModel, self).__init__(model, model_weight, model_bias, model_type, savedir, **kwargs)
    
    def reinitialize_model(self, time=None):
        def init_weights(m):
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(init_weights)

    def analyze_results(self, true, pred, pred_gp):
        true, pred = map(self.inverse_transform, (true, pred))
        pred_gp = self.inverse_transform(pred_gp) if pred_gp is not None else None
        return super().analyze_results(true, pred, pred_gp)