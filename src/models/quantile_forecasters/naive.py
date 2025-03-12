from src.models.quantile_forecasters.base import BaseQuantileForecaster
from torch.distributions import Normal
import torch
from torch import Tensor


class NaiveGaussian(BaseQuantileForecaster):
    """
    Naive Gaussian model predicts quantiles by Gaussian distribution with
    parameters mu and sigma, where mu is the last observation, sigma is the
    standard deviation of the one-step naive forecast's residuals multiplied by
    the  square root of the forecasting horizon. The model is based on Hyndman,
    R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice,
    3rd edition, Chapter 5.5, OTexts: Melbourne, Australia. OTexts.com/fpp3.
    Accessed on March, 2025.
    """

    @staticmethod
    def load_model(path: str) -> "NaiveGaussian":
        hparams = torch.load(path)
        return NaiveGaussian(**hparams)

    def _predict_quantiles(self, input_seq: Tensor) -> Tensor:
        loc = input_seq[:, :, [-1], None]
        loc = loc.repeat(1, 1, self.output_len, len(self.quantile_levels))
        residual = input_seq[:, :, 1:] - input_seq[:, :, :-1]
        std = (residual**2).mean(dim=2) ** 0.5
        std = std[:, :, None, None]
        std = std.repeat(1, 1, self.output_len, len(self.quantile_levels))
        sqrt_len = torch.arange(1, self.output_len + 1) ** 0.5
        std = std * sqrt_len[None, None, :, None]
        q = self.quantile_levels[None, None, None, :]
        q = q.repeat(1, loc.shape[1], self.output_len, 1)
        qvalues = Normal(loc, std).icdf(q)
        return qvalues


class NaiveBootstrap(BaseQuantileForecaster):
    """
    Naive Bootstrap model predicts quantiles by empirical quantiles of the
    cumulative sum of the last observation and the random residuals of the
    one-step naive forecast.  The model is based on Hyndman, R.J., &
    Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd
    edition, Chapter 5.5, OTexts: Melbourne, Australia. OTexts.com/fpp3.
    Accessed on March, 2025.
    """
    def __init__(
        self,
        input_len: int,
        output_len: int,
        target_dim: int,
        quantile_levels: list[float] = [0.025, 0.5, 0.975],
        step_size: int = 1,
        num_samples: int = 1000,
    ):
        """
        Args:
            input_len: length of the input sequence aka context length.
            output_len: length of the output sequence aka forecast horizon.
            quantile_levels: list of the quantiles to be forecasted.
            step_size: step size in the sliding window dataset.
            num_samples: number of sampled residuals is used for emprical
                quantile calculation.
        """
        super().__init__(input_len, output_len, target_dim, quantile_levels, step_size)
        self.save_hyperparameters()
        self.num_samples = num_samples

    @staticmethod
    def load_model(path: str) -> "NaiveBootstrap":
        hparams = torch.load(path)
        return NaiveBootstrap(**hparams)

    def _predict_quantiles(self, input_seq: Tensor) -> Tensor:
        b, _, size = input_seq.shape
        h = self.output_len
        residual = input_seq[:, :, 1:] - input_seq[:, :, :-1]
        forecast_sample = torch.zeros(b, self.target_dim, h, self.num_samples)
        for i in range(self.num_samples):
            resampled_residual = residual[:, :, torch.randint(0, size - 1, (h,))]
            last = input_seq[:, :, [-1]]
            forecast_sample[:, :, :, i] = last + torch.cumsum(resampled_residual, dim=2)
        qvalues = torch.quantile(forecast_sample, self.quantile_levels, dim=3)
        qvalues = qvalues.permute(1, 2, 3, 0)
        return qvalues
