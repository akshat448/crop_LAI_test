import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import functions

class DataModule(pl.LightningDataModule):
    def __init__(self, feat_path, adj_path, batch_size=32, seq_len=12, pre_len=3, split_ratio=0.8, normalize=True, scaling_method='minmax', **kwargs):
        super(DataModule, self).__init__()
        self.feat_path = feat_path
        self.adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.scaling_method = scaling_method

        # Load features and adjacency matrix
        self.feat = functions.load_features(self.feat_path)
        self._adj = functions.load_adjacency_matrix(self.adj_path)
        assert np.all(self._adj >= 0), "Adjacency matrix contains negative values"

        # Print the shapes of the feature matrix and adjacency matrix
        print("Feature matrix shape:", self.feat.shape)
        print("Adjacency matrix shape:", self._adj.shape)

        # Calculate min and max values of features for normalization
        self._feat_min_val = np.min(self.feat)
        self._feat_max_val = np.max(self.feat)
        print("Original feature range:", self._feat_min_val, self._feat_max_val)

        # Apply scaling if normalization is enabled
        if self.normalize:
            if self.scaling_method == 'minmax':
                self.feat = (self.feat - self._feat_min_val) / (self._feat_max_val - self._feat_min_val)
            elif self.scaling_method == 'std':
                self._feat_mean = np.mean(self.feat, axis=0)
                self._feat_std = np.std(self.feat, axis=0)
                self.feat = (self.feat - self._feat_mean) / self._feat_std
            else:
                raise ValueError("Invalid scaling method. Choose 'minmax' or 'std'.")
            
            print(f"Scaled feature range: {np.min(self.feat)}, {np.max(self.feat)}")

    @property
    def adj(self):
        return self._adj

    @property
    def feat_max_val(self):
        return self._feat_max_val

    def setup(self, stage=None):
        # Generate torch datasets
        train_dataset, val_dataset = functions.generate_torch_datasets(
            self.feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=False,  # Set to False because we've already normalized the data
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Print shapes of tensors in datasets
        for i, tensor in enumerate(self.train_dataset.tensors):
            print(f"Shape of tensor {i} in train dataset:", tensor.shape)

        for i, tensor in enumerate(self.val_dataset.tensors):
            print(f"Shape of tensor {i} in validation dataset:", tensor.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()