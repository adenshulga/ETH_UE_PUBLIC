import typing as tp
from src.evaluation.evaluator_abc import Evaluator, Results
from src.data_manipulation.custom_dataset_abc import SizedDataset
from torch import Tensor
import torch


class UncertaintyEvaluator(Evaluator):
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
        return Results(metrics=metrics, images=None)


class ECE(UncertaintyEvaluator):
    def evaluate(
        self,
        model_predictions: tp.Sequence[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        ece = []
        for i in range(len(self.quantiles) // 2):    
            lbound = model_predictions[:, :, i]
            ubound = model_predictions[:, :, -i-1]
            mask = (lbound <= test_dataset) & (test_dataset <= ubound)
            picp = mask.to(float).mean().item()
            ece.append(abs(picp - (self.quantiles[-i-1] - self.quantiles[i])))
        metrics = {'ece': sum(ece) / len(ece)}
        return Results(metrics=metrics, images=None)


class CRPS(UncertaintyEvaluator):
    def evaluate(
        self,
        model_predictions: tp.Sequence[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        pred = model_predictions
        target = test_dataset[:, :, None]
        error = torch.abs(pred - target) * 2
        umask = target < pred
        lmask = target >= pred
        q = torch.tensor(self.quantiles)
        q = q[None, None, :].repeat(pred.shape[0], pred.shape[1], 1)
        error[umask] = error[umask] * q[umask]
        error[lmask] = error[lmask] * (1 - q[lmask])
        metrics = {'crps': error.mean().item()}
        return Results(metrics=metrics, images=None)
