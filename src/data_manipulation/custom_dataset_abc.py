from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from src.common_entities.custom_exceptions import CustomException
import typing as tp
import numpy as np

"""
Desing principles:

- pytorch dataset/dataloader?

- strategy/model can accept different data modalities, however there will be a lot of 
different datasets for each modality. For e.g. btc/eth/sol time  series data, 
their orderbooks, options data, etc... I want to notate this input
- dataset for model consists of this data of different modalities, therefore 


"""

PricePoint = tuple[float, np.datetime64]


class SizedDataset[T](Dataset[T]):
    """Same as usual torch Dataset, but with enforcing of abstract methods"""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        pass


class SizedDatasetMapping[T](SizedDataset[T]):
    """
    Proxy class for writing simple train_test_split given the dataset.
    The problem is that dataset might be too big or too complex to split
    via pythonic methods via indices because it might be too slow
    """

    def __init__(
        self, indices_mapping: tp.Sequence[int], original_dataset: SizedDataset[T]
    ):
        self.indices_mapping = indices_mapping
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.indices_mapping)

    def __getitem__(self, index: int) -> T:
        return self.original_dataset[self.indices_mapping[index]]


HistoricPricesDataset = tp.NewType("HistoricPricesDataset", SizedDataset[PricePoint])
LastNPricesRegularDataset = tp.NewType(
    "LastNPricesRegularDataset", SizedDataset[Tensor]
)
LastNPricesUnRegularDataset = tp.NewType(
    "LastNPricesUnRegularDataset", SizedDataset[list[PricePoint]]
)


# class MultiModalDatasetError(CustomException):
#     pass


# class MultiModalDataset[*Ts](SizedDataset[tuple[*Ts]]):
#     """
#     A generic dataset that zips together an arbitrary number of modality-specific datasets.

#     Each dataset should be a Dataset[T] where T is the sample type for that modality.
#     __getitem__ returns a tuple (dataset1[i], dataset2[i], ...).

#     Due to the incompletness of type annotation features in modern python,
#     always specify this class generic params
#     """

#     def __init__(self, *datasets: SizedDataset[tp.Any]) -> None:
#         self.datasets = datasets
#         if not all(len(ds) == self.datasets[0] for ds in self.datasets):  # type: ignore
#             raise MultiModalDatasetError("datasets")
#         self._len = len(self.datasets[0])  # type: ignore

#     def __len__(self) -> int:
#         # return self._length
#         return self._len

#     def __getitem__(self, index: int) -> tuple[*Ts]:
#         return tuple(ds[index] for ds in self.datasets)
