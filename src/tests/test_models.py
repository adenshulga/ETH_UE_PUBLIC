import torch
from inspect import getmembers, isclass
import pytest
from src import models


model_classes = [m[1] for m in getmembers(models, isclass)]


class TestOnSyntheticData:
    def setup_class(self):
        self.input_len = 100
        self.output_len = 25
        self.quantiles = [0.1, 0.5, 0.9]
        self.train_dataset = torch.rand(10, 1000)
        self.test_dataset = torch.rand(10, 1000)
    
    @pytest.mark.parametrize('model_class', model_classes)
    def test_creation(self, model_class):
        model = model_class(self.input_len, self.output_len)

    @pytest.mark.parametrize('model_class', model_classes)
    def test_training(self, model_class):
        model = model_class(self.input_len, self.output_len)
        model.fit(self.train_dataset)

    @pytest.mark.parametrize('model_class', model_classes)
    def test_prediction(self, model_class):
        model = model_class(self.input_len, self.output_len)
        model.fit(self.train_dataset)
        pred = model.predict(self.test_dataset)
        assert pred.shape == (
            self.train_dataset.shape[0], self.output_len, len(self.quantiles)
        )
    
    @pytest.mark.parametrize('model_class', model_classes)
    def test_loading(self, model_class):
        model = model_class(self.input_len, self.output_len)
        path = model.save_model()
        model = model_class.load_model(path)
        assert model.input_len == self.input_len
        assert model.output_len == self.output_len
