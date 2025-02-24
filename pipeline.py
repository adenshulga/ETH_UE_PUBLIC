import os

import hydra
from hydra.utils import instantiate
from joblib import Memory
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger

from src.data_manipulation.custom_dataset_abc import SizedDataset
from src.data_manipulation.train_test_utils import sequential_train_test_split
from src.evaluation.evaluator_abc import Evaluator
from src.models.model_abc import CustomModel
from src.utils.env_var_loading import get_env_var

os.makedirs("data/cache", exist_ok=True)
# use memory.cache decorator in case of computationally expensive function
memory = Memory(location="./data/cache")

logger = CometLogger(
    api_key=get_env_var("COMET_API_KEY"),
    project_name="eth-ue",
    auto_output_logging=True,
)


# This square brackets syntax is "new generic syntax", helps specifying type variables
@hydra.main()
def main[ElementaryDataT, TransformedDataT, PredictedT](
    cfg: DictConfig,
) -> None:
    # NOTE: types should be defined explicitly only where their derivation is impossible
    # such in the case when object is constructed in runtime

    dataset: SizedDataset[ElementaryDataT] = instantiate(cfg["dataset"])
    # because type is defined in runtime

    model: CustomModel[ElementaryDataT, TransformedDataT, PredictedT] = instantiate(
        cfg["model"]
    )

    transformed_dataset: SizedDataset[TransformedDataT] = model.transform(dataset)

    train_dataset, test_dataset = sequential_train_test_split(
        transformed_dataset, train_ratio=0.7
    )

    model.fit(train_dataset)

    prediction = model.predict(test_dataset)

    # evaluation (It would be good to implement K-Fold crossvalidation later)

    results_evaluator: Evaluator = instantiate(cfg["evaluator"])
    results = results_evaluator.evaluate(prediction, test_dataset)
    results.log(logger)


if __name__ == "__main__":
    main()
