import typing as tp
from abc import abstractmethod, ABC

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset

from src.common_entities.custom_exceptions import CustomException
from omegaconf import DictConfig
from hydra.utils import instantiate

"""
Desing principles:

- pytorch dataset/dataloader?

- strategy/model can accept different data modalities, however there will be a lot of 
different datasets for each modality. For e.g. btc/eth/sol time  series data, 
their orderbooks, options data, etc... I want to notate this input
- dataset for model consists of this data of different modalities, therefore 


"""

PricePoint = tuple[np.datetime64, float]


# class SizedDataset[T](ABC, Dataset[T]):
class SizedDataset[T](tp.Protocol):
    """Same as usual torch Dataset, but with enforcing of abstract methods"""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T: ...

    def __add__(self, other: "SizedDataset[T]") -> "SizedDataset[T]":
        return ConcatDataset[T]([self, other])  # type: ignore
        # here type ignore is used because ConcatDataset is poorly typed


class SizedDatasetMapping[T](SizedDataset[T]):
    """
    Proxy class for writing simple train_test_split given the dataset.
    The problem is that dataset might be too big or too complex to split
    via pythonic methods via indices because it might be too slow
    """

    def __init__(
        self, indices_mapping: tp.Sequence[int], original_dataset: SizedDataset[T]
    ) -> None:
        self.indices_mapping = indices_mapping
        self.original_dataset = original_dataset

    def __len__(self) -> int:
        return len(self.indices_mapping)

    def __getitem__(self, index: int) -> T:
        return self.original_dataset[self.indices_mapping[index]]


class MultiModalDatasetError(CustomException):
    pass


class MultiModalDataset[*Ts](SizedDataset[tuple[*Ts]]):
    """
    A generic dataset that zips together an arbitrary number of modality-specific datasets.

    Each dataset should be a Dataset[T] where T is the sample type for that modality.
    __getitem__ returns a tuple (dataset1[i], dataset2[i], ...).

    Due to the incompletness of type annotation features in modern python,
    always specify this class generic params
    """

    def __init__(self, *datasets: SizedDataset[tp.Any]) -> None:
        self.datasets = datasets
        if not all(len(ds) == self.datasets[0] for ds in self.datasets):
            raise MultiModalDatasetError("datasets have different lengthes")
        self._len = len(self.datasets[0])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> tuple[*Ts]:
        return tuple(ds[index] for ds in self.datasets)

    @staticmethod
    def from_cfg(dataset_configs: list[DictConfig]) -> "MultiModalDataset[*Ts]":
        datasets: list[SizedDataset[tp.Any]] = []
        for cfg in dataset_configs:
            dataset: SizedDataset[tp.Any] = instantiate(cfg)
            datasets.append(dataset)

        return MultiModalDataset(*datasets)
