from src.evaluation import UncertaintyEvaluator
import torch


class TestOnSyntheticData:
    def setup_class(self):
        torch.manual_seed(1)
        self.test_dataset = torch.randn(10, 100)
        pred = torch.zeros(10, 100, 3)
        pred[:, :, 0] = pred[:, :, 0] - 2
        pred[:, :, 2] = pred[:, :, 2] + 2
        self.model_prediction = pred
        self.quantiles = [0.025, 0.5, 0.975]
    
    def test_picp(self):
        evaluator = UncertaintyEvaluator(quantiles=self.quantiles)
        res = evaluator.evaluate(self.model_prediction, self.test_dataset)
        assert round(res.metrics['picp'], 2) == (
            self.quantiles[-1] - self.quantiles[0]
        )
