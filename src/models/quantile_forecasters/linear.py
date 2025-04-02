from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster
import torch.nn as nn
import torch
from torch import Tensor


class LinearModule(nn.Module):
    def __init__(
        self, target_dim: int, num_quantiles: int
    ):
        super().__init__()
        self.target_dim = target_dim
        self.num_quantiles = num_quantiles
        self.linear = nn.Linear(target_dim, target_dim * num_quantiles)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, self.target_dim, self.num_quantiles)
        return x


class LinearForecaster(BaseTorchQuantileForecaster):
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

        self.model = LinearModule(
            target_dim=target_dim,
            num_quantiles=len(self.quantile_levels),
        )

    @staticmethod
    def load_model(path: str) -> "LinearForecaster":
        return LinearForecaster.load_from_checkpoint(path)

    def _predict_quantiles(self, input_seq: Tensor) -> Tensor:
        with torch.no_grad():
            qvalues = self.model(input_seq)
        qvalues = qvalues[:, -self.output_len:, :]
        return qvalues
