from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.distributions import Normal


def kernel(X1, X2, scale=1.0):
    cdist = torch.cdist(X1, X2, p=2) ** 2
    return torch.exp(-cdist / 2 / scale**2)


class GPModule(nn.Module):
    def __init__(
        self,
        target_dim,
        num_inducing: int,
        kernel_scale: float,
        quantile_levels: Tensor,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, target_dim))
        self.mean_linear = nn.Linear(num_inducing, target_dim)
        self.var_linear = nn.Linear(num_inducing, target_dim)
        self.kernel_scale = kernel_scale
        self.quantile_levels = quantile_levels
        self.target_dim = target_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        x = x.reshape(-1, self.target_dim)
        K_xz = kernel(x, self.inducing_points, self.kernel_scale)
        mean = self.mean_linear(K_xz)
        mean = mean[:, :, None]
        mean = mean.repeat(1, 1, len(self.quantile_levels))
        var = F.softplus(self.var_linear(K_xz)) + 1e-4
        var = var[:, :, None]
        var = var.repeat(1, 1, len(self.quantile_levels))
        q = self.quantile_levels[None, None, :]
        q = q.repeat(mean.shape[0], mean.shape[1], 1)
        qvalues = Normal(loc=mean, scale=var**0.5).icdf(q)
        qvalues = qvalues.reshape(
            batch_size, -1, self.target_dim, len(self.quantile_levels)
        )
        return qvalues


class GPForecaster(BaseTorchQuantileForecaster):
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
        num_inducing: int = 128,
        scale: float = 1.0,
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
            num_inducing: number of inducing poinst to train Gaussian process.
            scale: scale of the RBF kernel in Gaussian process
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
            False,
        )
        self.save_hyperparameters()
        self.num_inducing = num_inducing
        self.scale = scale

        self.model = GPModule(
            target_dim,
            num_inducing,
            scale,
            self.quantile_levels,
        )

    @staticmethod
    def load_model(path: str) -> "GPForecaster":
        return GPForecaster.load_from_checkpoint(path)
