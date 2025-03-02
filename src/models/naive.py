import typing as tp
from src.models.model_abc import CustomModel
from src.data_manipulation.custom_dataset_abc import LastNPricesRegularDataset
from torch.distributions import Normal
import torch


class NaiveGaussian(CustomModel):
    def __init__(
        self, 
        input_len: int, 
        output_len: int, 
        quantiles: list[float]=[0.1, 0.5, 0.9]
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.quantiles = quantiles

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return dataset
    
    def save_model(self):
        return ''
    
    @staticmethod
    def load_model(path):
        return NaiveGaussian(input_len=100, output_len=5)
        
    def predict(self, dataset):
        loc = dataset[:, -1]
        scale = dataset[:, -self.input_len:].std(dim=1)
        qvalues = []
        for i in range(1, self.output_len+1):
            gaussian = Normal(loc, scale*(i**0.5))
            qvalues_i = torch.stack([gaussian.icdf(torch.tensor(q)) for q in self.quantiles])
            qvalues.append(qvalues_i)
        qvalues = torch.stack(qvalues)
        qvalues = qvalues.permute(2, 0, 1)
        return qvalues
