from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster
import torch.nn as nn
import torch
from torch import Tensor


class MLPModule(nn.Module):
    def __init__(
        self, target_dim: int, hidden_dim: int, num_layers: int, num_quantiles: int
    ):
        super().__init__()
        self.target_dim = target_dim
        self.num_quantiles = num_quantiles
        layers = [nn.Linear(target_dim, hidden_dim), nn.ReLU()]
        for _ in range(1, num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, target_dim * num_quantiles))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = x.reshape(x.shape[0], -1, self.target_dim, self.num_quantiles)
        return x


class MLPForecaster(BaseTorchQuantileForecaster):
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
        logging: bool = False,
        prevent_crossing = False,
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
            logging: if True, enables logging in Comet ML in training.
            prevent_crossing: if True, prevents quantile crossing issue.
            num_layers: number of MLP layers.
            hidden_dim: hidden dimensionality of MLP layers.
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
            logging,
            prevent_crossing,
        )
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.model = MLPModule(
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_quantiles=len(self.quantile_levels),
        )

    @staticmethod
    def load_model(path: str) -> "MLPForecaster":
        return MLPForecaster.load_from_checkpoint(path)
