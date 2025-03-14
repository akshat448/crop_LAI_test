import torch
from torch import nn
from block import SpatialBlock


def get_device():
    # return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device("cpu")


class Model(nn.Module):
    def __init__(self, adj, seq_len, input_dim, hidden_dim, output_dim, num_nodes, num_stacks, num_layers):
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
        # Dropout Layer
        self.dropout = nn.Dropout(p=0.45)
        self.rnn_dropout = nn.Dropout(p=0.45)
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
            inputs = inputs.unsqueeze(1).repeat(1, seq_len, 1, 1)  # Add seq_len dimension
            inputs = inputs.reshape(batch_size * seq_len, num_nodes, -1)  # Flatten for GRU input

        # Ensure correct input to GRU
        if inputs.size(-1) != self.gru_input_size:
            inputs = nn.Linear(inputs.size(-1), self.gru_input_size).to(device)(inputs)

        temporal_out, _ = self.temporal_block(inputs)  # GRU expects [batch_size, seq_len, feature_dim]
        temporal_out = self.rnn_dropout(temporal_out)  # Apply dropout to the output of the GRU

        # Output layer
        final_output = self.final_layer(temporal_out)

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
