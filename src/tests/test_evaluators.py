from src.evaluation.uncertainty_evaluator import PICP, ECE, CRPS
import torch


class TestOnSyntheticData:
    def setup_class(self):
        torch.manual_seed(1)
        self.test_dataset = torch.randn(1, 10, 100)
        pred = torch.zeros(1, 10, 100, 5)
        pred[:, :, :, 0] = -2
        pred[:, :, :, 1] = -1
        pred[:, :, :, 2] = 0
        pred[:, :, :, 3] = 1
        pred[:, :, :, 4] = 2
        self.model_prediction = pred
        self.quantiles = [0.025, 0.16, 0.5, 0.84, 0.975]
    
    def test_picp(self):
        evaluator = PICP(quantiles=self.quantiles)
        res = evaluator.evaluate(self.model_prediction, self.test_dataset)
        assert round(res.metrics['picp'], 2) == (
            self.quantiles[-1] - self.quantiles[0]
        )

    def test_ece(self):
        evaluator = ECE(quantiles=self.quantiles)
        res = evaluator.evaluate(self.model_prediction, self.test_dataset)
        assert round(res.metrics['ece'], 2) == 0
    
    def test_crps(self):
        evaluator = CRPS(quantiles=self.quantiles)
        res = evaluator.evaluate(self.model_prediction, self.test_dataset)
        assert round(res.metrics['crps'], 2) == 2.48
