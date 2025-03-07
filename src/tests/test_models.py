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
        self.train_seq = torch.rand(10, 1000)
        self.test_seq = torch.rand(10, 1000)
    
    @pytest.mark.parametrize('model_class', model_classes)
    def test_creation(self, model_class):
        model = model_class(self.input_len, self.output_len)
    
    @pytest.mark.parametrize('model_class', model_classes)
    def test_transform(self, model_class):
        model = model_class(self.input_len, self.output_len)
        train_dataset = model.transform(self.train_seq)

    @pytest.mark.parametrize('model_class', model_classes)
    def test_training(self, model_class):
        model = model_class(self.input_len, self.output_len)
        train_dataset = model.transform(self.train_seq)
        model.fit(train_dataset)

    @pytest.mark.parametrize('model_class', model_classes)
    def test_prediction(self, model_class):
        model = model_class(self.input_len, self.output_len)
        train_dataset = model.transform(self.train_seq)
        test_dataset = model.transform(self.test_seq)
        model.fit(train_dataset)
        for (input_seq, target_seq) in test_dataset:
            break
        input_seq = input_seq[None, ...]
        pred = model.predict(input_seq)
        assert pred.shape == (
            1, self.train_seq.shape[0], 
            self.output_len, len(self.quantiles)
        )

    @pytest.mark.parametrize('model_class', model_classes)
    def test_loading(self, model_class):
        model = model_class(self.input_len, self.output_len)
        path = model.save_model()
        model = model_class.load_model(path)
        assert model.input_len == self.input_len
        assert model.output_len == self.output_len
