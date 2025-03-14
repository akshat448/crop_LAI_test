# import torch
# import torch.nn as nn
# from utils.utils import (
#     calculate_normalized_laplacian,
#     calculate_random_walk_matrix,
#     calculate_scaled_laplacian
# )
# from functions import WaveletTransform

# def get_device():
#     #return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#     return torch.device("cpu")

# class TGCN_Conv(nn.Module):
#     def __init__(self, num_nodes, hidden_dim, output_dim):
#         super(TGCN_Conv, self).__init__()
#         self.num_nodes = num_nodes
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.W0 = None
#         self.W1 = nn.Parameter(torch.randn(hidden_dim, output_dim, dtype=torch.float32))
#         self.biases = nn.Parameter(torch.zeros(1, num_nodes, output_dim, dtype=torch.float32))

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W1)
#         nn.init.constant_(self.biases, 0.0)
    
#     def forward(self, inputs, laplacian):
#         device = get_device()
#         inputs, laplacian = inputs.to(device, dtype=torch.float32), laplacian.to(device, dtype=torch.float32)

#         batch_size, seq_len, num_nodes = inputs.shape
#         assert num_nodes == self.num_nodes, "Mismatch in number of nodes"

#         if self.W0 is None or self.W0.size(0) != seq_len:
#             self.W0 = nn.Parameter(torch.randn(seq_len, self.hidden_dim, device=device, dtype=torch.float32))
#             nn.init.xavier_uniform_(self.W0)

#         inputs_reshaped = inputs.permute(0, 2, 1)  # [batch_size, num_nodes, seq_len]
#         a_times_inputs = torch.einsum("ij,bjt->bit", laplacian, inputs_reshaped)  # [batch_size, num_nodes, seq_len]
        
#         a_times_inputs = a_times_inputs.reshape(batch_size * num_nodes, seq_len)  # [batch_size * num_nodes, seq_len]
        
#         a_times_inputs = torch.matmul(a_times_inputs, self.W0)  # [batch_size * num_nodes, hidden_dim]
        
#         a_times_inputs = a_times_inputs.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, hidden_dim]
#         a_times_inputs = torch.relu(a_times_inputs)
        
#         a_times_inputs = torch.matmul(a_times_inputs, self.W1)  # [batch_size, num_nodes, output_dim]
        
#         a_times_inputs += self.biases

#         return a_times_inputs
# class SpatialBlock(nn.Module):
#     def __init__(self, adj, seq_len, input_dim, hidden_dim, output_dim, adaptive_adj=True):
#         super(SpatialBlock, self).__init__()
#         self.device = get_device()  # Assign device to self.device
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.num_nodes = adj.shape[0]
#         self.seq_len = seq_len  # Define seq_len

#         self.distance_laplacian = calculate_normalized_laplacian(adj).to(dtype=torch.float32, device=self.device)
#         self.random_walk_matrix = calculate_random_walk_matrix(adj).to(dtype=torch.float32, device=self.device)
#         self.scaled_laplacian = calculate_scaled_laplacian(adj).to(dtype=torch.float32, device=self.device)

#         self.graph_convs = nn.ModuleList([
#             TGCN_Conv(self.num_nodes, self.hidden_dim, self.output_dim).to(self.device) for _ in range(3)
#         ])

#         self.wavelet = WaveletTransform(level=1).to(self.device)  # Minimal reduction

#         self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0, 1.0]).to(self.device))

#         self.residual = nn.Linear(input_dim, output_dim).to(self.device)  # Adjusted to match output_dim

#         self.W = nn.Parameter(torch.randn(self.num_nodes, hidden_dim, device=self.device))

#         self.spatial_proj = nn.Linear(hidden_dim, output_dim).to(self.device)


#     def calculate_spatial_similarity(self, wavelet_low_freq):
        
#         mean = wavelet_low_freq.mean(dim=1, keepdim=True)  # (batch_size, 1, num_nodes)
#         diff = mean.unsqueeze(-1) - mean.unsqueeze(-2)  # (batch_size, num_nodes, num_nodes)
#         return torch.exp(-torch.norm(diff, dim=1))  # Gaussian kernel for similarity

#     def forward(self, inputs):
#         device = self.device
#         inputs = inputs.to(device)
#         batch_size, seq_len, num_nodes = inputs.shape
        
#         # Wavelet decomposition
#         wavelet_low, _ = self.wavelet(inputs)  # Low-frequency components: [batch_size, seq_len, num_nodes]

#         wavelet_low = torch.nn.functional.interpolate(
#             wavelet_low.permute(0, 2, 1),
#             size=seq_len,
#             mode='linear',
#             align_corners=False
#         ).permute(0, 2, 1)  # Ensure it is back to [batch_size, seq_len, num_nodes]

#         # Calculate spatial similarity matrix
#         spatial_similarity = self.calculate_spatial_similarity(wavelet_low)  # [batch_size, num_nodes, num_nodes]

#         # Apply graph convolutions
#         graph_outputs = []
#         for idx, laplacian in enumerate([self.distance_laplacian, self.random_walk_matrix, self.scaled_laplacian]):
            
#             assert laplacian.shape[0] == laplacian.shape[1] == num_nodes, "Laplacian dimensions do not match num_nodes"
#             graph_out = self.graph_convs[idx](wavelet_low, laplacian)
#             graph_outputs.append(graph_out)

#         # Project spatial_similarity to hidden_dim using W
#         spatial_similarity = spatial_similarity.to(device)
#         spatial_similarity_projected = torch.einsum(
#             "bij,jk->bik", spatial_similarity, self.W
#         )  # [batch_size, num_nodes, hidden_dim]

#         # Reshape to expected shape for output
#         spatial_similarity_projected = spatial_similarity_projected.permute(0, 2, 1)  # [batch_size, hidden_dim, num_nodes]
#         spatial_similarity_projected = spatial_similarity_projected.permute(0, 2, 1)  # [batch_size, num_nodes, hidden_dim]
        
#         # Project spatial similarity to output_dim
#         spatial_similarity_projected = self.spatial_proj(spatial_similarity_projected)  # [batch_size, num_nodes, output_dim]
#         # Combine graph outputs and spatial similarity
#         weights = torch.softmax(self.weights, dim=0)  # Normalize weights

#         combined_output = (
#             weights[0] * graph_outputs[0] +
#             weights[1] * graph_outputs[1] +
#             weights[2] * graph_outputs[2] +
#             weights[3] * spatial_similarity_projected
#         )

#         # Add residual connection
#         residual = self.residual(wavelet_low)  # [batch_size, seq_len, output_dim]
#         residual = torch.nn.functional.interpolate(
#             residual.permute(0, 2, 1),  # Interpolate along the temporal dimension
#             size=num_nodes,  # Match num_nodes
#             mode='linear',
#             align_corners=False
#         ).permute(0, 2, 1)  # [batch_size, num_nodes, output_dim]
#         combined_output += residual
#         #print(f"After operation, combined_output shape: {combined_output.shape}")
#         return combined_output


import torch
import torch.nn as nn
from utils.utils import (
    calculate_normalized_laplacian,
    calculate_random_walk_matrix,
    calculate_scaled_laplacian
)
from functions import WaveletTransform

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TGCN_Conv(nn.Module):
    def __init__(self, num_nodes, hidden_dim, output_dim):
        super(TGCN_Conv, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W0 = None
        self.W1 = nn.Parameter(torch.randn(hidden_dim, output_dim, dtype=torch.float32))
        self.biases = nn.Parameter(torch.zeros(1, num_nodes, output_dim, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.constant_(self.biases, 0.0)
    
    def forward(self, inputs, laplacian):
        device = get_device()
        inputs, laplacian = inputs.to(device, dtype=torch.float32), laplacian.to(device, dtype=torch.float32)

        batch_size, seq_len, num_nodes = inputs.shape
        assert num_nodes == self.num_nodes, "Mismatch in number of nodes"

        if self.W0 is None or self.W0.size(0) != seq_len:
            self.W0 = nn.Parameter(torch.randn(seq_len, self.hidden_dim, device=device, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W0)

        inputs_reshaped = inputs.permute(0, 2, 1)  # [batch_size, num_nodes, seq_len]
        a_times_inputs = torch.einsum("ij,bjt->bit", laplacian, inputs_reshaped)  # [batch_size, num_nodes, seq_len]
        
        a_times_inputs = a_times_inputs.reshape(batch_size * num_nodes, seq_len)  # [batch_size * num_nodes, seq_len]
        
        a_times_inputs = torch.matmul(a_times_inputs, self.W0)  # [batch_size * num_nodes, hidden_dim]
        
        a_times_inputs = a_times_inputs.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, hidden_dim]
        a_times_inputs = torch.relu(a_times_inputs)
        
        a_times_inputs = torch.matmul(a_times_inputs, self.W1)  # [batch_size, num_nodes, output_dim]
        
        a_times_inputs += self.biases

        return a_times_inputs


class SpatialBlock(nn.Module):
    def __init__(self, adj, seq_len, input_dim, hidden_dim, output_dim, adaptive_adj=True):
        super(SpatialBlock, self).__init__()
        self.device = get_device()  # Assign device to self.device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = adj.shape[0]
        self.seq_len = seq_len  # Define seq_len

        self.distance_laplacian = calculate_normalized_laplacian(adj).to(dtype=torch.float32, device=self.device)
        self.random_walk_matrix = calculate_random_walk_matrix(adj).to(dtype=torch.float32, device=self.device)
        self.scaled_laplacian = calculate_scaled_laplacian(adj).to(dtype=torch.float32, device=self.device)

        self.graph_convs = nn.ModuleList([
            TGCN_Conv(self.num_nodes, self.hidden_dim, self.output_dim).to(self.device) for _ in range(3)
        ])

        self.wavelet = WaveletTransform(level=2).to(self.device)  # Minimal reduction

        self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0, 1.0]).to(self.device))

        self.residual = nn.Linear(input_dim, output_dim).to(self.device)  # Adjusted to match output_dim

        self.W = nn.Parameter(torch.randn(self.num_nodes, hidden_dim, device=self.device))

        self.spatial_proj = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self.similarity_threshold = 0.2  # Added (prevents noisy connections)
        self.temp = nn.Parameter(torch.tensor(1.5))  # Initialize higher

    def calculate_spatial_similarity(self, wavelet_low_freq):
        mean = wavelet_low_freq.mean(dim=1, keepdim=True)  # (batch_size, 1, num_nodes)
        diff = mean.unsqueeze(-1) - mean.unsqueeze(-2)  # (batch_size, num_nodes, num_nodes)
        similarity = torch.exp(-torch.norm(diff, dim=1) / self.temp)  # Gaussian kernel for similarity
        similarity = torch.where(similarity > self.similarity_threshold, similarity, torch.zeros_like(similarity))  # Apply threshold
        return similarity

    def forward(self, inputs):
        device = self.device
        inputs = inputs.to(device)
        batch_size, seq_len, num_nodes = inputs.shape
        
        # Wavelet decomposition
        wavelet_low, _ = self.wavelet(inputs)  # Low-frequency components: [batch_size, seq_len, num_nodes]

        wavelet_low = torch.nn.functional.interpolate(
            wavelet_low.permute(0, 2, 1),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # Ensure it is back to [batch_size, seq_len, num_nodes]

        # Calculate spatial similarity matrix
        spatial_similarity = self.calculate_spatial_similarity(wavelet_low)  # [batch_size, num_nodes, num_nodes]

        # Apply graph convolutions
        graph_outputs = []
        for idx, laplacian in enumerate([self.distance_laplacian, self.random_walk_matrix, self.scaled_laplacian]):
            assert laplacian.shape[0] == laplacian.shape[1] == num_nodes, "Laplacian dimensions do not match num_nodes"
            graph_out = self.graph_convs[idx](wavelet_low, laplacian)
            graph_outputs.append(graph_out)

        # Project spatial_similarity to hidden_dim using W
        spatial_similarity = spatial_similarity.to(device)
        spatial_similarity_projected = torch.einsum(
            "bij,jk->bik", spatial_similarity, self.W
        )  # [batch_size, num_nodes, hidden_dim]

        # Reshape to expected shape for output
        spatial_similarity_projected = spatial_similarity_projected.permute(0, 2, 1)  # [batch_size, hidden_dim, num_nodes]
        spatial_similarity_projected = spatial_similarity_projected.permute(0, 2, 1)  # [batch_size, num_nodes, hidden_dim]
        
        # Project spatial similarity to output_dim
        spatial_similarity_projected = self.spatial_proj(spatial_similarity_projected)  # [batch_size, num_nodes, output_dim]
        # Combine graph outputs and spatial similarity
        weights = torch.softmax(self.weights, dim=0)  # Normalize weights

        combined_output = (
            weights[0] * graph_outputs[0] +
            weights[1] * graph_outputs[1] +
            weights[2] * graph_outputs[2] +
            weights[3] * spatial_similarity_projected
        )

        # Apply LayerNorm to the last dimension (output_dim)
        combined_output = self.layer_norm(combined_output)

        # Add residual connection
        residual = self.residual(wavelet_low)  # [batch_size, seq_len, output_dim]
        residual = torch.nn.functional.interpolate(
            residual.permute(0, 2, 1),  # Interpolate along the temporal dimension
            size=num_nodes,  # Match num_nodes
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [batch_size, num_nodes, output_dim]

        # Add skip connection normalization:
        combined_output = combined_output + 0.3 * residual  # Weighted residual
        return combined_output