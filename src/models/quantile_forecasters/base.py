from src.models.model_abc import CustomModel
from src.data_manipulation.custom_dataset_abc import SizedDataset
from src.data_manipulation.sliding_window import SlidingWindowDataset
from torch import Tensor
import torch
from pytorch_lightning import LightningModule
from abc import abstractmethod
import typing as tp


class BaseQuantileForecaster(CustomModel, LightningModule):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        quantile_levels: list[float] = [0.025, 0.5, 0.975],
        step_size: int = 1,
    ):
        """
        Args:
            input_len: length of the input sequence aka context length.
            output_len: length of the output sequence aka forecast horizon.
            quantile_levels: list of the quantiles to be forecasted.
            step_size: step size in the sliding window dataset.
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_len = input_len
        self.output_len = output_len
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
