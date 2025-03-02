from src.evaluation.evaluator_abc import Evaluator, Results


class UncertaintyEvaluator(Evaluator):
    def __init__(self, quantiles: list[float]):
        self.quantiles = quantiles
    
    def evaluate(self, model_predictions, test_dataset):
        lbound = model_predictions[:, :, 0]
        ubound = model_predictions[:, :, -1]
        mask = (lbound <= test_dataset) & (test_dataset <= ubound)
        picp = mask.to(float).mean().item()
        metrics = {'picp': picp}
        res = Results(metrics=metrics, images=None)
        return res
