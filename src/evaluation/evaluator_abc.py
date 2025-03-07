from abc import ABC, abstractmethod
import typing as tp
from dataclasses import dataclass
from src.data_manipulation.custom_dataset_abc import SizedDataset
from pytorch_lightning.loggers import CometLogger


@dataclass
class Results:
    metrics: dict[str, float] | None
    images: tp.Sequence[str] | None  # paths with plots

    def log(self, logger: CometLogger) -> None:
        if self.metrics is not None:
            logger.log_metrics(self.metrics)
        if self.images is not None:
            [logger.experiment.log_image(img) for img in self.images]


@dataclass
class Evaluator[PredictedT, TransformedDataT](ABC):
    """
    - In future i want to save all metrics and results, push them to comet_ml
    - Single evaluator instance - single pipeline run
    """

    @abstractmethod
    def evaluate(
        self,
        model_predictions: SizedDataset[PredictedT],
        test_dataset: SizedDataset[TransformedDataT],
    ) -> Results:
        pass
