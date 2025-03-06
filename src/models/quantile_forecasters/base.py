import typing as tp
from src.models.model_abc import CustomModel
from abc import abstractmethod, ABC
from src.data_manipulation.custom_dataset_abc import SizedDataset
from src.data_manipulation.sliding_window import SlidingWindowDataset
from torch import Tensor
from pytorch_lightning import LightningModule


class BaseQuantileForecaster(CustomModel, LightningModule):
    def __init__(
        self, 
        input_len: int, 
        output_len: int, 
        quantiles: list[float]=[0.025, 0.5, 0.975],
        step_size: int=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_len = input_len
        self.output_len = output_len
        self.quantiles = quantiles
        self.step_size = step_size

    def transform(
        self, dataset: SizedDataset[Tensor]
    ) -> SizedDataset[SlidingWindowDataset]:
        return SlidingWindowDataset(
            sequence=dataset,
            window_size=self.input_len,
            step_size=self.step_size,
            shift_size=self.output_len,
        )
