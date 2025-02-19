from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from src.common_entities.custom_exceptions import CustomException
import typing as tp
import numpy as np

"""
Desing principles:
- 



- pytorch dataset/dataloader?

- strategy/model can accept different data modalities, however there will be a lot of 
different datasets for each modality. For e.g. btc/eth/sol time  series data, 
their orderbooks, options data, etc... I want to notate this input
- dataset for model consists of this data of different modalities, therefore 


"""

PricePoint = tuple[float, np.datetime64]


class HistoricPricesDataset(Dataset[PricePoint]):
    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> PricePoint:
        pass


class LastNPricesRegularDataset(Dataset[Tensor]):
    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> tuple[Tensor]:
        pass


class LastNPricesUnRegularDataset(Dataset[list[PricePoint]]):
    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> list[PricePoint]:
        pass


class MultiModalDatasetError(CustomException):
    pass


class MultiModalDataset[*Ts](Dataset[tuple[*Ts]]):
    """
    A generic dataset that zips together an arbitrary number of modality-specific datasets.

    Each dataset should be a Dataset[T] where T is the sample type for that modality.
    __getitem__ returns a tuple (dataset1[i], dataset2[i], ...).

    Due to the incompletness of type annotation features in modern python,
    always specify this class generic params
    """

    def __init__(self, *datasets: Dataset[tp.Any]) -> None:
        self.datasets = datasets
        if not all(len(ds) == self.datasets[0] for ds in self.datasets):  # type: ignore
            raise MultiModalDatasetError("datasets")
        self._len = len(self.datasets[0])  # type: ignore

    def __len__(self) -> int:
        # return self._length
        return self._len

    def __getitem__(self, index: int) -> tuple[*Ts]:
        return tuple(ds[index] for ds in self.datasets)
