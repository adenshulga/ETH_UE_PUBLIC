from src.evaluation.evaluator_abc import Evaluator, Results
from src.data_manipulation.custom_dataset_abc import SizedDataset
from abc import abstractmethod
from torch import Tensor
import torch


class UncertaintyEvaluator(Evaluator):
    def __init__(self, quantiles: list[float]) -> None:
        self.quantiles = quantiles
    
    def evaluate(
        self,
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> Results:
        """
        Args:
            model_predictions: predicted quantiles of the shape (b, d, l, q), 
                where b is the batch size, d is the dimensionality, h is the 
                forecast horizon, q is the number of quantiles.
            test_dataset: input sequence of the shape (b, d, l), where b is the
                batch size, d is the dimensionality, h is the sequence length.
        Returns:
            Results: results with attributes metrics and images.
        """
        pass
        metrics = self._calculate_metrics(model_predictions, test_dataset)
        return Results(metrics=metrics, images=None)
    
    @abstractmethod
    def _calculate_metrics(self, model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> dict:
        pass


class PICP(UncertaintyEvaluator):
    def _calculate_metrics(
        self,
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> dict:
        pred = torch.tensor(model_predictions)
        target = torch.tensor(test_dataset)
        lbound = pred[:, :, :, 0]
        ubound = pred[:, :, :, -1]
        mask = (lbound <= target) & (target <= ubound)
        picp = mask.float().mean().item()
        return {'picp': picp}


class ECE(UncertaintyEvaluator):
    def _calculate_metrics(
        self,
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> dict:
        pred = torch.tensor(model_predictions)
        target = torch.tensor(test_dataset)
        ece = []
        for i in range(len(self.quantiles) // 2):
            lbound = pred[:, :, :, i]
            ubound = pred[:, :, :, -i-1]
            mask = (lbound <= target) & (target <= ubound)
            picp = mask.float().mean().item()
            ece.append(abs(picp - (self.quantiles[-i-1] - self.quantiles[i])))
        return {'ece': sum(ece) / len(ece)}


class CRPS(UncertaintyEvaluator):
    def _calculate_metrics(
        self,
        model_predictions: SizedDataset[Tensor],
        test_dataset: SizedDataset[Tensor],
    ) -> dict:
        pred = torch.tensor(model_predictions)
        target = torch.tensor(test_dataset)[:, :, :, None]
        error = torch.abs(pred - target) * 2
        umask = target < pred
        lmask = target >= pred
        q = torch.tensor(self.quantiles)[None, None, None, :]
        q = q.repeat(1, pred.shape[1], pred.shape[2], 1)
        error[umask] = error[umask] * q[umask]
        error[lmask] = error[lmask] * (1 - q[lmask])
        return {'crps': error.mean().item()}
