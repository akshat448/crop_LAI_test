import numpy as np
import pandas as pd
import torch
import pywt
from pywt import Wavelet
from torch import nn

def load_features(feat_path):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df)
    return feat

def load_adjacency_matrix(adj_path):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df)
    return adj

def generate_dataset(data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True):
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val

    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_X, train_Y, test_X, test_Y = [], [], [], []

    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))

    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))

    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

def generate_torch_datasets(data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset

# class WaveletTransform(nn.Module):
#     def __init__(self, wavelet='db1', mode='symmetric', level=None):
#         super(WaveletTransform, self).__init__()
#         self.wavelet = wavelet
#         self.mode = mode
#         self.level = level

#     def forward(self, inputs):
#         batch_size, seq_len, num_nodes = inputs.shape
#         approx_coeff_list = []
#         detail_coeff_list = []

#         for node_data in inputs.permute(2, 0, 1):  # Iterate over nodes
#             coeffs = pywt.wavedec(node_data.detach().cpu().numpy(), self.wavelet, mode=self.mode, level=self.level)
#             approx_coeff_list.append(coeffs[0])  # Approximation coefficients
#             if len(coeffs) > 1:
#                 detail_coeff_list.append(coeffs[1:])  # Detail coefficients
#             else:
#                 detail_coeff_list.append(np.zeros_like(coeffs[0]))  # Add zeros if detail coeffs are missing

#         # Convert the lists to tensors
#         approx_coeff = torch.tensor(np.stack(approx_coeff_list), device=inputs.device)
#         detail_coeff = torch.tensor(np.stack(detail_coeff_list), device=inputs.device)

#         # Reshape back to original dimensions
#         approx_coeff = approx_coeff.view(batch_size, -1, num_nodes)  # [B, T_approx, N]
#         detail_coeff = detail_coeff.view(batch_size, -1, num_nodes)  # [B, T_detail, N]

#         return approx_coeff, detail_coeff
    
# class WaveletTransform(nn.Module):
#     def __init__(self, wavelet='db1', mode='symmetric', level=None):
#         super(WaveletTransform, self).__init__()
#         self.wavelet = wavelet
#         self.mode = mode
#         self.level = level

#     def forward(self, inputs):
#         # Check the shape of inputs
#         batch_size, seq_len, num_nodes = inputs.shape  # Assuming a [B, T, N] shape

#         # Reshape to process each node independently
#         inputs = inputs.permute(0, 2, 1)  # [B, N, T]
#         inputs_reshaped = inputs.reshape(-1, seq_len)  # [(B * N), T]

#         # Determine the maximum allowable level
#         max_level = pywt.dwt_max_level(seq_len, Wavelet(self.wavelet).dec_len)
#         level = self.level if self.level is not None else max_level
#         level = min(level, max_level)  # Ensure the level is within allowable range

#         # Perform wavelet decomposition on each node's time-series data
#         approx_coeff_list = []
#         detail_coeff_list = []
#         for node_data in inputs_reshaped:
#             # Detach the tensor from the computation graph before converting to NumPy
#             coeffs = pywt.wavedec(node_data.detach().cpu().numpy(), self.wavelet, mode=self.mode, level=level)
#             approx_coeff_list.append(coeffs[0])  # Approximation coefficients
#             if len(coeffs) > 1:
#                 detail_coeff_list.append(coeffs[1:])  # Detail coefficients
#             else:
#                 detail_coeff_list.append(np.zeros_like(coeffs[0]))  # Add zeros if detail coeffs are missing

#             # Convert the lists to tensors
#         approx_coeff = torch.tensor(np.stack(approx_coeff_list), device=inputs.device)
#         detail_coeff = torch.tensor(np.stack(detail_coeff_list), device=inputs.device)

#         # Reshape back to original dimensions
#         approx_coeff = approx_coeff.view(batch_size, -1, num_nodes)  # [B, T_approx, N]
#         detail_coeff = detail_coeff.view(batch_size, -1, num_nodes)  # [B, T_detail, N]

#         return approx_coeff, detail_coeff


from pytorch_wavelets import DWT1DForward

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db1', mode='symmetric', level=1):
        super(WaveletTransform, self).__init__()
        self.dwt = DWT1DForward(wave=wavelet, J=level, mode=mode)

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_len, num_nodes]
        inputs = inputs.permute(0, 2, 1)  # [batch_size, num_nodes, seq_len]
        approx, coeffs = self.dwt(inputs)  # [batch_size, num_nodes, T_approx]
        approx = approx.permute(0, 2, 1)  # [batch_size, T_approx, num_nodes]
        return approx, coeffs