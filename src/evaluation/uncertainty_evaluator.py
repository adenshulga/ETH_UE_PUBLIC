import typing as tp
from src.evaluation.evaluator_abc import Evaluator, Results
from abc import ABC
from src.data_manipulation.custom_dataset_abc import SizedDataset
from torch import Tensor


class UncertaintyEvaluator(Evaluator, ABC):
    def __init__(self, quantiles: list[float]) -> None:
        self.quantiles = quantiles


class PICP(UncertaintyEvaluator):
    def evaluate(
        self,
        model_predictions: tp.Sequence[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        lbound = model_predictions[:, :, 0]
        ubound = model_predictions[:, :, -1]
        mask = (lbound <= test_dataset) & (test_dataset <= ubound)
        picp = mask.to(float).mean().item()
        metrics = {'picp': picp}
        res = Results(metrics=metrics, images=None)
        return res


class ECE(UncertaintyEvaluator):
    def evaluate(self, model_predictions, test_dataset):
        ece = []
        for i in range(len(self.quantiles) // 2):    
            lbound = model_predictions[:, :, i]
            ubound = model_predictions[:, :, -i-1]
            mask = (lbound <= test_dataset) & (test_dataset <= ubound)
            picp = mask.to(float).mean().item()
            ece.append(abs(picp - (self.quantiles[-i-1] - self.quantiles[i])))
        metrics = {'ece': sum(ece) / len(ece)}
        res = Results(metrics=metrics, images=None)
        return res
