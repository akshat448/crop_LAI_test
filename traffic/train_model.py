import torch_optimizer as optim
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def custom_loss(y_pred, y_true, l1_weight=0.004):
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    mae = torch.mean(torch.abs(y_pred - y_true))
    return 0.2 * rmse + 0.8 * mae + l1_weight

class TrainModel(pl.LightningModule):
    def __init__(self, model: nn.Module, pre_len: int = 3, learning_rate: float = 0.001, feat_max_val: float = 1.0, feat_min_val: float = 0.0, l1_weight=0.001, scaling_method='minmax', augment=True, **kwargs):
        super(TrainModel, self).__init__()
        self.save_hyperparameters("pre_len", "learning_rate", "feat_max_val", "feat_min_val", "scaling_method")
        self.model = model
        self.pre_len = pre_len
        self.learning_rate = learning_rate
        self.feat_max_val = feat_max_val
        self.feat_min_val = feat_min_val
        self.l1_weight = l1_weight
        self.scaling_method = scaling_method
        self.augment = augment

        self.rmse_metric = MeanSquaredError(squared=False)
        self.mae_metric = MeanAbsoluteError()
        self.validation_outputs = []

        if self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'std':
            self.scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'minmax' or 'std'.")

    def forward(self, x):
        predictions = self.model(x)[:, -self.pre_len:, :, 0]
        return predictions

    def shared_step(self, batch):
        x, y = batch
        predictions = self(x)
        y = y.view(x.shape[0], self.pre_len, x.shape[2])
        return predictions, y

    def augment_data(self, x):
        noise = torch.randn_like(x) * 0.02
        x += noise
        # jitter = torch.normal(0, 0.02, size=x.shape)
        # jitter = torch.cumsum(jitter, dim=1)  # Correlated noise
        # x += jitter
        return x

    def compute_loss(self, predictions, targets):
        return custom_loss(predictions, targets, self.l1_weight)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.augment:
            x = self.augment_data(x)
        predictions, y = self.shared_step((x, y))
        loss = self.compute_loss(predictions, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch)

        if self.scaling_method == 'std' and not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(y.view(-1, y.shape[-1]))

        if self.scaling_method == 'minmax':
            predictions_rescaled = predictions * (self.feat_max_val - self.feat_min_val) + self.feat_min_val
            y_rescaled = y * (self.feat_max_val - self.feat_min_val) + self.feat_min_val
        elif self.scaling_method == 'std':
            predictions_rescaled = self.scaler.inverse_transform(predictions.view(-1, predictions.shape[-1])).view(predictions.shape)
            y_rescaled = self.scaler.inverse_transform(y.view(-1, y.shape[-1])).view(y.shape)

        loss = self.compute_loss(predictions_rescaled, y_rescaled)
        self.rmse_metric.update(predictions_rescaled.view(-1), y_rescaled.view(-1))
        self.mae_metric.update(predictions_rescaled.view(-1), y_rescaled.view(-1))

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_rmse = self.rmse_metric.compute()
        avg_mae = self.mae_metric.compute()

        self.log("val_rmse", avg_rmse, on_epoch=True, prog_bar=True)
        self.log("val_mae", avg_mae, on_epoch=True, prog_bar=True)

        self.rmse_metric.reset()
        self.mae_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=6e-4, betas=(0.95, 0.998))
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mae",
                "interval": "epoch",
                "frequency": 1,
            }
        }