from src.models.model_abc import CustomModel
from src.data_manipulation.custom_dataset_abc import SizedDataset
from src.data_manipulation.sliding_window import SlidingWindowDataset
from src.utils.env_var_loading import get_env_var
from torch import Tensor
from torch.optim import Optimizer
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
import os
from abc import abstractmethod
import typing as tp


def quantile_loss(pred: Tensor, target: Tensor, q: Tensor):
    """
    Args:
        pred: predicted quantile values of the shape (b, d, l, q)
        target: target values of the shape (b, d, l)
        q: quantile levels of the shape (q,)
    """
    q = q[None, None, None, :]
    target = target[..., None]
    upper = F.relu(pred - target) * (1 - q)
    lower = F.relu(target - pred) * q
    loss = 2 * (upper + lower).sum(dim=(1, 2, 3))
    return loss.mean()


class BaseQuantileForecaster(CustomModel, LightningModule):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        target_dim: int,
        quantile_levels: list[float] = [0.025, 0.5, 0.975],
        step_size: int = 1,
    ):
        """
        Args:
            input_len: length of the input sequence aka context length.
            output_len: length of the output sequence aka forecast horizon.
            target_dim: dimensionality of the target sequence
            quantile_levels: list of the quantiles to be forecasted.
            step_size: step size in the sliding window dataset.
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_len = input_len
        self.output_len = output_len
        self.target_dim = target_dim
        self.quantile_levels = torch.tensor(quantile_levels)
        self.step_size = step_size

    def transform(self, dataset: SizedDataset[Tensor]) -> SlidingWindowDataset:
        return SlidingWindowDataset(
            sequence=dataset,
            window_size=self.input_len,
            step_size=self.step_size,
            shift_size=self.output_len,
        )

    def predict(self, dataset: SizedDataset[Tensor]) -> SizedDataset[Tensor]:
        """
        Args:
            dataset: input sequence of the shape (b, d, l), where b is the
                batch size, d is the dimensionality, l is the sequence length.
        Returns:
            SizedDataset[Tensor]: predicted quantile values of the shape
                (b, d, h, q), where b is the batch size, d is the
                dimensionality, h is the forecast horizon, q is the number of
                quantiles.
        """
        input_seq = torch.tensor(dataset)[:, :, -self.input_len :]
        predicted_quantiles = self._predict_quantiles(input_seq)
        predicted_quantiles = tp.cast(SizedDataset[Tensor], predicted_quantiles)
        return predicted_quantiles

    @abstractmethod
    def _predict_quantiles(self, input_seq: Tensor) -> Tensor:
        pass

    def save_model(self) -> str:
        save_dir = ".eth-ue"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "hparams.pth")
        torch.save(dict(self.hparams), path)
        return path

    def fit(self, dataset: SizedDataset) -> None:
        pass


class BaseTorchQuantileForecaster(BaseQuantileForecaster):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        target_dim: int,
        quantile_levels: list[float] = [0.025, 0.5, 0.975],
        step_size: int = 1,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        accelerator: str = "cpu",
        enable_progress_bar: bool = True,
        logging: bool = False,
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
        """
        super().__init__(input_len, output_len, target_dim, quantile_levels, step_size)
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self.logging = logging
        self.model = nn.Module()

    def fit(self, dataset: SizedDataset) -> None:
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        logger = False
        if self.logging:
            logger = CometLogger(
                get_env_var("COMET_API_KEY"),
                project_name=".eth-ue",
            )
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=int(len(dataloader) * 0.1),
            logger=logger,
            accelerator=self.accelerator,
        )
        self.trainer.fit(
            model=self,
            train_dataloaders=dataloader,
        )
        self.eval()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self.calc_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def calc_loss(self, batch: Tensor) -> Tensor:
        input_seq, target_seq = batch
        qvalues = self.model(input_seq)
        loss = quantile_loss(qvalues, target_seq, self.quantile_levels)
        return loss

    def save_model(self) -> str:
        save_dir = ".eth-ue"
        checkpoint_path = os.path.join(save_dir, "model_checkpoint.ckpt")
        self.trainer.save_checkpoint(checkpoint_path)
        return checkpoint_path
