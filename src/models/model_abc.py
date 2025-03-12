import typing as tp
from abc import ABC, abstractmethod

from torch.nn import Module

from src.data_manipulation.custom_dataset_abc import SizedDataset


class CustomModel[ElementaryDataT, TransformedDataT, PredictedT](Module, ABC):
    """
    Logging should be done via comet ml, with passing CometLogger to pl.Trainer
    """

    @abstractmethod
    def transform(
        self, dataset: SizedDataset[ElementaryDataT]
    ) -> SizedDataset[TransformedDataT]:
        pass

    @abstractmethod
    def fit(self, dataset: SizedDataset[TransformedDataT]) -> None:
        """
        Returns data describing training process,
        for e.g. gradient norms from epoch to epoch, losses and etc if necessary.
        """
        pass

    @abstractmethod
    def predict(
        self, dataset: SizedDataset[TransformedDataT]
    ) -> SizedDataset[PredictedT]:
        pass

    @abstractmethod
    def save_model(self) -> str:
        """Returns path where model is saved"""
        pass
    
    @staticmethod
    @abstractmethod
    def load_model(path: str) -> "CustomModel":
        """Loads model given path"""
        pass


tp.Sequence
