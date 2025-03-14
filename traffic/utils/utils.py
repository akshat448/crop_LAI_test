import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
from scipy.sparse import coo_matrix
from scipy.sparse import linalg
import torch

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


"""def calculate_normalized_laplacian(adj):
    assert (adj >= 0).all(), "Adjacency matrix contains negative values before conversion to sparse matrix"
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    adj_dense = adj.toarray()
    d_inv_sqrt = np.power(d + 1e-10, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_laplacian = normalized_laplacian.maximum(0)  # Ensure all values are non-negative
    return normalized_laplacian"""

def calculate_normalized_laplacian(adj):
    if torch.is_tensor(adj):
        device = adj.device
        adj = adj.detach().cpu().numpy()
    else:
        device = 'cpu'
    adj = np.maximum(adj, 0)
    assert (adj >= 0).all(), "Adjacency matrix contains negative values before conversion to sparse matrix"
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d + 1e-5, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_laplacian = normalized_laplacian.maximum(0)
    return torch.tensor(normalized_laplacian.toarray(), dtype=torch.float32, device=device)


def calculate_random_walk_matrix(adj_mx):
    # Ensure the adjacency matrix is a dense tensor before further operations
    if isinstance(adj_mx, torch.Tensor):
        adj_mx = adj_mx.cpu().numpy()
    
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(d, -1)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = coo_matrix((d_inv, (np.arange(d_inv.shape[0]), np.arange(d_inv.shape[0]))), shape=adj_mx.shape)
    random_walk_mx = d_mat_inv.dot(adj_mx)

    # Convert to tensor and move to appropriate device
    random_walk_tensor = torch.tensor(random_walk_mx, dtype=torch.float32)
    return random_walk_tensor


def calculate_reverse_random_walk_matrix(adj_mx):
    # Ensure the adjacency matrix is a dense tensor before further operations
    if isinstance(adj_mx, torch.Tensor):
        adj_mx = adj_mx.cpu().numpy()  # Convert to NumPy if it's a tensor
    
    # Calculate the degree matrix inverse
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(d, -1)
    d_inv[np.isinf(d_inv)] = 0.
    
    # Create a diagonal matrix from the inverse degree values
    d_mat_inv = np.diag(d_inv)
    
    # Compute the reverse random walk matrix by multiplying the degree matrix with the transposed adjacency matrix
    reverse_random_walk_mx = np.dot(d_mat_inv, adj_mx.T)
    
    # Convert the resulting matrix back to a PyTorch tensor
    reverse_random_walk_tensor = torch.tensor(reverse_random_walk_mx, dtype=torch.float32)
    
    return reverse_random_walk_tensor



def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        if isinstance(adj_mx, torch.Tensor):
            adj_mx_np = np.maximum.reduce([adj_mx.cpu().detach().numpy(), adj_mx.cpu().detach().numpy().T])
        else:
            adj_mx_np = np.maximum.reduce([adj_mx, adj_mx.T])
    else:
        if isinstance(adj_mx, torch.Tensor):
            adj_mx_np = adj_mx.cpu().detach().numpy()
        else:
            adj_mx_np = adj_mx

    L = calculate_normalized_laplacian(adj_mx_np)

    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I

    # Convert back to tensor and send to the same device as adj_mx
    if isinstance(adj_mx, torch.Tensor):
        return torch.tensor(L.toarray(), dtype=torch.float32).to(adj_mx.device)
    else:
        return torch.tensor(L.toarray(), dtype=torch.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data