import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.functional import normalize

# calculating the normalized laplacian matrix
# L = D^(-1/2) * A * D^(-1/2)
def calculate_laplacian(matrix):
    device = matrix.device  # Ensure the laplacian is calculated on the same device as the input matrix
    
    # adding identity matrix to the adjacency matrix (self-loops)
    matrix = matrix + torch.eye(matrix.size(0)).to(device)
    
    # sum of each row (edges per node)
    row_sum = matrix.sum(1)
    
    # normalizing the matrix
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    
    # replacing infinities with 0
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    # creating the diagonal matrix
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # normalizing the matrix
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    
    return normalized_laplacian
