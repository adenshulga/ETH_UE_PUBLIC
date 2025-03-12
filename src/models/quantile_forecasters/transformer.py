from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster
import torch
import torch.nn as nn
from torch import Tensor


class TransformerEncoderModule(nn.Module):
    def __init__(
        self,
        target_dim: int,
        d_model: int,
        dim_feedforward: int,
        num_layers: int,
        num_heads: int,
        num_quantiles: int,
        dropout: float,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.num_quantiles = num_quantiles
        self.input_projection = nn.Linear(target_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_projection = nn.Linear(
            in_features=d_model, out_features=target_dim * num_quantiles
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], self.target_dim, -1, self.num_quantiles)
        return x


class TransfomerForecaster(BaseTorchQuantileForecaster):
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
        d_model: int = 32,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
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
            d_model: the number of expected features in the input of Transformer.
            nhead: the number of heads in Transformer.
            dim_feedforward: the dimension of the feedforward network model of 
                Transformer.
            dropout: the dropout value of Transformer.
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
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.model = TransformerEncoderModule(
            target_dim=target_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_heads=num_heads,
            num_quantiles=len(self.quantile_levels),
            dropout=dropout,
        )

    @staticmethod
    def load_model(path: str) -> "TransfomerForecaster":
        return TransfomerForecaster.load_from_checkpoint(path)

    def _predict_quantiles(self, input_seq: Tensor) -> Tensor:
        with torch.no_grad():
            qvalues = self.model(input_seq)
        qvalues = qvalues[:, :, -self.output_len :, :]
        return qvalues
