import torch
from inspect import getmembers, isclass
import pytest
from src import models
from src.models.quantile_forecasters.base import BaseTorchQuantileForecaster


model_classes = [m[1] for m in getmembers(models, isclass)]


class TestOnSyntheticData:
    def setup_class(self):
        self.input_len = 100
        self.output_len = 25
        self.quantile_levels = [0.1, 0.5, 0.9]
        self.target_dim = 10
        self.num_epochs = 1
        self.batch_size = 10000
        self.train_seq = torch.rand(self.batch_size, self.target_dim)
        self.test_seq = torch.rand(self.batch_size, self.target_dim)
        

    def create_model(self, model_class):
        if issubclass(model_class, BaseTorchQuantileForecaster):
            model = model_class(
                self.input_len, 
                self.output_len, 
                self.target_dim,
                self.quantile_levels, 
                num_epochs=self.num_epochs, 
                batch_size=self.batch_size
            )
        else:
            model = model_class(self.input_len, self.output_len, self.target_dim, self.quantile_levels)
        return model
    
    @pytest.mark.parametrize('model_class', model_classes)
    def test_creation(self, model_class):
        model = self.create_model(model_class)
    
    @pytest.mark.parametrize('model_class', model_classes)
    def test_transform(self, model_class):
        model = self.create_model(model_class)
        train_dataset = model.transform(self.train_seq)
        input_seq, target_seq = train_dataset[0]
        assert input_seq.shape == (self.input_len, self.target_dim)
        assert target_seq.shape == (self.input_len, self.target_dim)

    @pytest.mark.parametrize('model_class', model_classes)
    def test_training(self, model_class):
        model = self.create_model(model_class)
        train_dataset = model.transform(self.train_seq)
        model.fit(train_dataset)

    @pytest.mark.parametrize('model_class', model_classes)
    def test_prediction(self, model_class):
        model = self.create_model(model_class)
        test_dataset = model.transform(self.test_seq)
        for (input_seq, _) in test_dataset:
            input_seq = input_seq[None, ...]
            pred = model.predict(input_seq)
            assert pred.shape == (
                1, 
                self.output_len,
                self.target_dim,
                len(self.quantile_levels),
            )
            break

    @pytest.mark.parametrize('model_class', model_classes)
    def test_loading(self, model_class):
        model = self.create_model(model_class)
        train_dataset = model.transform(self.train_seq)
        model.fit(train_dataset)
        path = model.save_model()
        model = model_class.load_model(path)
        assert model.input_len == self.input_len
        assert model.output_len == self.output_len
