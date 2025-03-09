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
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        pred = torch.tensor(model_predictions)
        target = torch.tensor(test_dataset)
        lbound = pred[:, :, :, 0]
        ubound = pred[:, :, :, -1]
        mask = (lbound <= target) & (target <= ubound)
        picp = mask.float().mean().item()
        metrics = {'picp': picp}
        return Results(metrics=metrics, images=None)


class ECE(UncertaintyEvaluator):
    def evaluate(
        self,
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        pred = torch.tensor(model_predictions)
        target = torch.tensor(test_dataset)
        ece = []
        for i in range(len(self.quantiles) // 2):
            lbound = pred[:, :, :, i]
            ubound = pred[:, :, :, -i-1]
            mask = (lbound <= target) & (target <= ubound)
            picp = mask.float().mean().item()
            ece.append(abs(picp - (self.quantiles[-i-1] - self.quantiles[i])))
        metrics = {'ece': sum(ece) / len(ece)}
        return Results(metrics=metrics, images=None)


class CRPS(UncertaintyEvaluator):
    def evaluate(
        self,
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        pred = torch.tensor(model_predictions)
        target = torch.tensor(test_dataset)[:, :, :, None]
        error = torch.abs(pred - target) * 2
        umask = target < pred
        lmask = target >= pred
        q = torch.tensor(self.quantiles)[None, None, None, :]
        q = q.repeat(1, pred.shape[1], pred.shape[2], 1)
        error[umask] = error[umask] * q[umask]
        error[lmask] = error[lmask] * (1 - q[lmask])
        metrics = {'crps': error.mean().item()}
        return Results(metrics=metrics, images=None)
