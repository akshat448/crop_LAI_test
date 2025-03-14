# import torch
# import torch.nn as nn
# from .base import ModelBase

# class ProbSparseSelfAttention(nn.Module):
#     def __init__(self, d_model, n_heads, factor=5):
#         super(ProbSparseSelfAttention, self).__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.factor = factor
#         self.qkv = nn.Linear(d_model, d_model * 3)
#         self.fc = nn.Linear(d_model, d_model)

#     def forward(self, query, key=None):
#         B, N, C = query.shape
#         if key is None:
#             key = query
#         qkv = self.qkv(query).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * (C // self.n_heads) ** -0.5
#         attn = attn.softmax(dim=-1)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.fc(out)
#         return out

# class InformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
#         super(InformerEncoderLayer, self).__init__()
#         self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Linear(d_ff, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         attn_out = self.self_attn(x)
#         x = self.norm1(x + self.dropout(attn_out))
#         ffn_out = self.ffn(x)
#         x = self.norm2(x + self.dropout(ffn_out))
#         return x

# class InformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.15):
#         super(InformerDecoderLayer, self).__init__()
#         self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
#         self.cross_attn = ProbSparseSelfAttention(d_model, n_heads)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Linear(d_ff, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, memory):
#         self_attn_out = self.self_attn(x)
#         x = self.norm1(x + self.dropout(self_attn_out))
#         cross_attn_out = self.cross_attn(x, memory)
#         x = self.norm2(x + self.dropout(cross_attn_out))
#         ffn_out = self.ffn(x)
#         x = self.norm3(x + self.dropout(ffn_out))
#         return x

# class Informer(nn.Module):
#     def __init__(self, input_dim, d_model=64, n_heads=2, d_ff=256, num_encoder_layers=2, num_decoder_layers=1, dropout=0.25):
#         super(Informer, self).__init__()
#         self.input_proj = nn.Linear(input_dim, d_model)
#         self.encoder_layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
#         self.decoder_layers = nn.ModuleList([InformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
#         self.output_proj = nn.Linear(d_model, 1)

#     def forward(self, x, return_last_dense=False):
#         x = self.input_proj(x)
#         for layer in self.encoder_layers:
#             x = layer(x)
#         memory = x
#         for layer in self.decoder_layers:
#             x = layer(x, memory)
#         if return_last_dense:
#             return x.mean(dim=1), x.view(x.size(0), -1)
#         x = x.mean(dim=1)
#         x = self.output_proj(x)
#         return x

# class InformerModel(ModelBase):
#     def __init__(self, input_dim, savedir, **kwargs):
#         model = Informer(input_dim=input_dim)
#         model_weight = "output_proj.weight"
#         model_bias = "output_proj.bias"
#         model_type = "informer"
#         self.device = kwargs.get('device', torch.device("cpu"))
#         super(InformerModel, self).__init__(model, model_weight, model_bias, model_type, savedir, **kwargs)
    
#     def reinitialize_model(self, time=None):
#         def init_weights(m):
#             if isinstance(m, (nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#         self.model.apply(init_weights)

#     def analyze_results(self, true, pred, pred_gp):
#         return ModelBase.analyze_results(self, true, pred, pred_gp)


import torch
import torch.nn as nn
import numpy as np
import math
from .base import ModelBase

class ConvLayer(nn.Module):
    """Distilling convolution layer with max pooling"""
    def __init__(self, d_model):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular'),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x.permute(0, 2, 1)  # [batch_size, d_model, seq_len]
        x = self.down_conv(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len//2, d_model]
        return self.norm(self.activation(x))

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5, attention_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads
        
        # Better weight initialization for stability
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = attention_dropout
        # Add dropout layer - this was missing
        self.dropout = nn.Dropout(attention_dropout)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.d_k)


    #   THIS WORKS GOOD FOR THE ATTENTION LAYER
    # def forward(self, query, key=None, value=None):
    #     if key is None:
    #         key = query
    #     if value is None:
    #         value = key
            
    #     B, N, C = query.size()
        
    #     # Standard multi-head attention
    #     qkv = self.qkv(query).view(B, N, 3, self.n_heads, C // self.n_heads)
    #     qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, d_k]
    #     q, k, v = qkv[0], qkv[1], qkv[2]  # each is [B, h, N, d_k]
        
    #     # Calculate attention scores with stable scaling
    #     attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
    #     # Apply softmax with improved numerical stability
    #     attn = torch.nn.functional.softmax(attn, dim=-1)
        
    #     # Apply attention dropout
    #     attn = torch.nn.functional.dropout(attn, p=self.attention_dropout, training=self.training)
        
    #     # Apply attention weights to values
    #     out = torch.matmul(attn, v)
        
    #     # Reshape and combine heads
    #     out = out.transpose(1, 2).contiguous().view(B, N, -1)
        
    #     return self.out_proj(out)
    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = key
            
        B, N, C = query.size()
        
        # Calculate Q, K, V
        qkv = self.qkv(query).view(B, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [B, h, N, d_k]
        
        # For each query, calculate its L_k most similar keys
        attn_all = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Calculate max and mean for each query across all keys
        M = torch.max(attn_all, dim=-1)[0] - torch.mean(attn_all, dim=-1)
        
        # Sample the top queries (using a safe percentage)
        n_top = max(1, min(int(N * 0.25), N-1))  # ensure at least 1 query
        M_top = M.topk(n_top, dim=-1, sorted=False)[1]
        
        # Create sparse mask for attention
        sparse_mask = torch.ones_like(attn_all, dtype=torch.bool)
        for b in range(B):
            for h in range(self.n_heads):
                sparse_mask[b, h, M_top[b, h]] = False
        
        # Apply mask (set unimportant scores to -inf)
        attn = attn_all.masked_fill(sparse_mask, -1e9)  # Use finite value for stability
        
        # Apply softmax and dropout
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # Now this will work with the dropout layer added
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and combine heads
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        
        return self.out_proj(out)
    
# class ProbSparseSelfAttention(nn.Module):
#     def __init__(self, d_model, n_heads, factor=5):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.factor = factor
#         self.d_k = d_model // n_heads
        
#         self.qkv = nn.Linear(d_model, d_model * 3)
#         self.out_proj = nn.Linear(d_model, d_model)
        
#     def _prob_sparse_attention(self, query, key, value):
#         B, L_Q, _ = query.size()
#         _, L_K, _ = key.size()

#         # Use standard attention without sparsity for now
#         # This ensures the output has the same dimension as input
#         attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
#         attn = torch.softmax(attn, dim=-1)
#         context = torch.matmul(attn, value)
        
#         return context, attn

#     def forward(self, query, key=None, value=None):
#         if key is None:
#             key = query
#         if value is None:
#             value = key
            
#         B, N, C = query.size()
        
#         qkv = self.qkv(query).view(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, N, d_k]
        
#         # Reshape for attention
#         q = q.transpose(1, 2).contiguous().view(B, N, -1)
#         k = k.transpose(1, 2).contiguous().view(B, N, -1)
#         v = v.transpose(1, 2).contiguous().view(B, N, -1)
        
#         context, attn = self._prob_sparse_attention(q, k, v)
#         context = context.reshape(B, N, self.d_model)
        
#         return self.out_proj(context)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, distil=True):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
        self.distil = distil
        self.conv = ConvLayer(d_model) if distil else None
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sparse attention
        attn_out = self.self_attn(x)

        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        # Distilling
        if self.distil and self.conv is not None:
            x = self.conv(x)
        return x

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Self-attention 
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads)
        # Cross-attention (between decoder and encoder)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # Self attention
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross attention
        cross_attn_out = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x

class Informer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, d_ff=512, 
                 num_encoder_layers=4, num_decoder_layers=2, dropout=0.2, distil=True):
        # Increased model capacity with larger d_model, more heads, and wider feed-forward layer
        super().__init__()
        self.distil = distil
        
        # Input projection with layer normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Encoder with increased layers
        self.encoder = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, distil=(distil and i < num_encoder_layers-1))
            for i in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection with multiple layers for better prediction
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.feature_dim = d_model
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, return_last_dense=False):
        B, seq_len, feature_dim = x.shape
        
        # Check for NaN inputs
        if torch.isnan(x).any():
            print("Warning: NaN values in input")
            # Replace NaNs with zeros
            x = torch.nan_to_num(x)
        
        # Input embedding
        x = self.input_proj(x)
        
        # Encoder
        for i, layer in enumerate(self.encoder):
            x_prev = x  # Store previous value for residual fallback
            try:
                x = layer(x)
                # Check for explosion or NaNs
                if torch.isnan(x).any() or torch.max(torch.abs(x)) > 1e6:
                    print(f"Warning: Unstable values in encoder layer {i}")
                    x = x_prev  # Use previous stable value
            except Exception as e:
                print(f"Error in encoder layer {i}: {e}")
                x = x_prev
        
        memory = x
        
        # Decoder with similar safeguards
        for i, layer in enumerate(self.decoder_layers):
            x_prev = x
            try:
                x = layer(x, memory)
                if torch.isnan(x).any() or torch.max(torch.abs(x)) > 1e6:
                    print(f"Warning: Unstable values in decoder layer {i}")
                    x = x_prev
            except Exception as e:
                print(f"Error in decoder layer {i}: {e}")
                x = x_prev
        
        # Global pooling for final prediction
        x_pooled = x.mean(dim=1)
        
        if return_last_dense:
            # For GP, we need to return both the prediction and the feature vector
            out = self.output_proj(x_pooled)
            # Use the flattened features as the feature vector for GP
            return out, x.reshape(B, -1)
        
        # Regular forward pass just returns the prediction
        return self.output_proj(x_pooled)

class InformerModel(ModelBase):
    def __init__(self, input_dim, savedir, use_gp=True, sigma=1, r_loc=0.5, r_year=1.5, 
                 sigma_e=0.32, sigma_b=0.01, use_sparse_gp=False, num_inducing=100, 
                 sparse_method='fitc', device=torch.device("cpu")):
        """
        Informer model for crop yield prediction
        """
        # Create the Informer model with appropriate architecture
        model = Informer(
            input_dim=input_dim,
            d_model=64,         # Model dimension
            n_heads=4,          # Number of attention heads
            d_ff=256,           # Feed-forward dimension
            num_encoder_layers=3,
            num_decoder_layers=2,
            dropout=0.2,
            distil=True         # Use distillation in encoder
        )
        
        # Fix: Change these to reference the correct layers in the Sequential module
        model_weight = "output_proj.3.weight"  # Point to the final linear layer's weight
        model_bias = "output_proj.3.bias"      # Point to the final linear layer's bias
        model_type = "informer"
        self.device = device
        
        # Initialize base model
        super().__init__(
            model, model_weight, model_bias, model_type, savedir,
            use_gp=use_gp, sigma=sigma, r_loc=r_loc, r_year=r_year, 
            sigma_e=sigma_e, sigma_b=sigma_b, use_sparse_gp=use_sparse_gp,
            num_inducing=num_inducing, sparse_method=sparse_method,
            device=device
        )
    
    def reinitialize_model(self, time=None):
        """
        Reinitialize the model weights for a new training run
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.model.apply(init_weights)
        
    def analyze_results(self, true, pred, pred_gp):
        """
        Analyze prediction results and compute metrics
        """
        return ModelBase.analyze_results(self, true, pred, pred_gp)
    
    def _normalize(self, train_images, val_images):
        """
        Properly handles normalization for time series data
        """
        # For time series, we need to handle normalization differently
        # First standardize by mean/std across the training set
        mean = np.mean(train_images, axis=0, keepdims=True)
        std = np.std(train_images, axis=0, keepdims=True) + 1e-8
        
        train_norm = (train_images - mean) / std
        val_norm = (val_images - mean) / std
        
        self.min_val = np.min(train_norm, axis=0, keepdims=True)
        self.max_val = np.max(train_norm, axis=0, keepdims=True)
        
        return train_norm, val_norm