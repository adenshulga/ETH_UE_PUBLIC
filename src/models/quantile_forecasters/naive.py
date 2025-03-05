import typing as tp
from src.models.quantile_forecasters.base import BaseQuantileForecaster
from src.data_manipulation.custom_dataset_abc import SizedDataset
from src.data_manipulation.sliding_window import SlidingWindowDataset
from torch.distributions import Normal
import torch
from torch import Tensor
import os


class NaiveGaussian(BaseQuantileForecaster):

    def fit(self, dataset: SizedDataset[SlidingWindowDataset]) -> None:
        pass
    
    def save_model(self) -> str:
        save_dir = '.temp'
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'hparams.pth')
        torch.save(dict(self.hparams), path)
        return path
    
    @staticmethod
    def load_model(path: str) -> "NaiveGaussian":
        hparams = torch.load(path)
        loaded_model = NaiveGaussian(**hparams)
        return loaded_model

    def predict(
        self, dataset: SizedDataset[Tensor]
    ) -> tp.Sequence[Tensor]:
        loc = dataset[:, [-1], None].repeat(1, self.output_len, len(self.quantiles))
        scale = dataset[:, -self.input_len:].std(dim=1)
        scale = scale[:, None, None].repeat(1, self.output_len, len(self.quantiles))
        sqrt_len = torch.arange(1, self.output_len + 1)**0.5
        scale = scale * sqrt_len[None, :, None]
        q = torch.tensor(self.quantiles)[None, None, :]
        q = q.repeat(len(loc), self.output_len, 1)
        qvalues = Normal(loc, scale).icdf(q)
        return qvalues
