from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster
import torch.nn as nn
import torch
from torch import Tensor


class LSTMModule(nn.Module):
    def __init__(
        self, target_dim: int, hidden_dim: int, num_layers: int, num_quantiles: int
    ):
        super().__init__()
        self.target_dim = target_dim
        self.num_quantiles = num_quantiles
        self.lstm = nn.LSTM(
            input_size=target_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            in_features=hidden_dim, out_features=target_dim * num_quantiles
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.output_projection(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], self.target_dim, -1, self.num_quantiles)
        return x


class LSTMForecaster(BaseTorchQuantileForecaster):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        target_dim: int,
        quantile_levels: list[float] = [0.025, 0.5, 0.975],
        step_size: int = 1,
        batch_size: int = 10000,
        num_epochs: int = 1,
        lr: float = 0.001,
        accelerator: str = "cpu",
        enable_progress_bar: bool = True,
        num_layers: int = 1,
        hidden_dim: int = 32,
    ):
        """
        Args:
            input_len: length of the input sequence aka context length.
            output_len: length of the output sequence aka forecast horizon.
            target_dim: dimensionality of the target sequence
            quantile_levels: list of the quantiles to be forecasted.
            step_size: step size in the sliding window dataset.
            batch_size: batch size in training.
            num_epochs: number of epochs in training.
            lr: learning rate in training.
            accelerator: name of the device for training.
            enable_progress_bar: if True, enables progress bar in training.
            num_layers: number of LSTM layers.
            hidden_dim: hidden dimensionality of LSTM layers.
        """
        super().__init__(
            input_len,
            output_len,
            target_dim,
            quantile_levels,
            step_size,
            batch_size,
            num_epochs,
            lr,
            accelerator,
            enable_progress_bar,
        )
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.model = LSTMModule(
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_quantiles=len(self.quantile_levels),
        )

    @staticmethod
    def load_model(path: str) -> "LSTMForecaster":
        return LSTMForecaster.load_from_checkpoint(path)

    def _predict_quantiles(self, input_seq: Tensor) -> Tensor:
        with torch.no_grad():
            qvalues = self.model(input_seq)
        qvalues = qvalues[:, :, -self.output_len :, :]
        return qvalues
