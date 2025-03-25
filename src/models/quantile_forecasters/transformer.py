from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros((1, max_len, d_model))
        position = torch.arange(max_len)[:, None]
        div_term = 1 / max_len ** (torch.arange(0, self.d_model, 2) / self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


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
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_projection = nn.Linear(
            in_features=d_model, out_features=target_dim * num_quantiles
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        x = x.reshape(x.shape[0], -1, self.target_dim, self.num_quantiles)
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
        logging: bool = False,
        prevent_crossing = False,
        num_layers: int = 1,
        d_model: int = 32,
        num_heads: int = 1,
        dim_feedforward: int = 64,
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
            logging: if True, enables logging in Comet ML in training.
            prevent_crossing: if True, prevents quantile crossing issue.
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
            logging,
            prevent_crossing,
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

