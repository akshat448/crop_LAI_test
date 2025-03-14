import torch
from torch import nn
from block import SpatialBlock


def get_device():
    return torch.device("cpu")

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5))
        attention_output = torch.matmul(attention_weights, V)
        return attention_output

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        head_dim = hidden_dim // self.num_heads

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        attention_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5))
        attention_output = torch.matmul(attention_weights, V).transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_dim)
        return self.out_proj(attention_output) + x  # Add residual connection


class Model(nn.Module):
    def __init__(self, adj, seq_len, input_dim, hidden_dim, output_dim, num_nodes, num_stacks, num_layers, num_heads):
        super(Model, self).__init__()
        self.num_stacks = num_stacks
        self.hidden_dim = hidden_dim
        self.gru_input_size = hidden_dim
        self.device = get_device()

        # Spatial Blocks
        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(adj, seq_len, input_dim, hidden_dim, output_dim).to(self.device) for _ in range(num_stacks)
        ])
        # Temporal Block (GRU)
        self.temporal_block = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True).to(self.device)
        # Attention Layer
        #self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.attention = SelfAttention(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Dropout Layer
        self.rnn_dropout = nn.Dropout(p=0.45)
        self.dropout = nn.Dropout(p=0.7)
        self.final_dropout = nn.Dropout(p=0.5)
        #self.layer_norm = nn.LayerNorm(hidden_dim)
        # Final Output Layer
        self.final_layer = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, inputs):
        device = self.device
        inputs = inputs.to(device, dtype=torch.float32)
        batch_size, seq_len, num_nodes = inputs.shape

        for stack in range(len(self.spatial_blocks)):

            # Reshape inputs to match expected shape for each SpatialBlock
            if inputs.shape != (batch_size, seq_len, num_nodes):
                inputs = inputs.permute(0, 2, 1)  # [batch_size, num_nodes, seq_len]

            inputs = self.spatial_blocks[stack](inputs)
            inputs = self.dropout(inputs)  # Apply dropout to the output of each SpatialBlock

        # Restore sequence length dimension
        if inputs.ndim == 3:  # Check if seq_len is missing
            inputs = inputs.permute(0, 2, 1)  # [batch_size, output_dim, num_nodes]
            inputs = inputs.unsqueeze(1).expand(-1, seq_len, -1, -1)  # Add seq_len dimension
            inputs = inputs.reshape(batch_size * seq_len, num_nodes, -1)  # Flatten for GRU input

        # Ensure correct input to GRU
        if inputs.size(-1) != self.gru_input_size:
            inputs = nn.Linear(inputs.size(-1), self.gru_input_size).to(device)(inputs)

        temporal_out, _ = self.temporal_block(inputs)  # GRU expects [batch_size, seq_len, feature_dim]
        temporal_out = self.rnn_dropout(temporal_out)  # Apply dropout to the output of the GRU
        #temporal_out = temporal_out + inputs  # Add residual connection
        gate = torch.sigmoid(nn.Linear(self.hidden_dim, 1)(temporal_out))
        temporal_out = gate * temporal_out + (1-gate) * inputs
        
        # Apply attention mechanism
        attention_out = self.attention(temporal_out)
        attention_out = self.layer_norm(attention_out + temporal_out)  # Add residual connection
        # Output layer
        final_output = self.final_layer(attention_out)
        final_output = self.final_dropout(final_output)
        #final_output = self.final_layer(temporal_out)
        # Reshape to match output dimensions
        output_dim = self.final_layer.out_features  # This should be 3

        final_output = final_output.view(batch_size, seq_len, num_nodes, output_dim)

        return final_output

    @property
    def hyperparameters(self):
        return {
            "input_dim": self.spatial_blocks[0].input_dim,
            "hidden_dim": self.spatial_blocks[0].hidden_dim,
            "output_dim": self.spatial_blocks[0].output_dim,
        }