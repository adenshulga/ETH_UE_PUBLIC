from src.data_manipulation.custom_dataset_abc import SizedDataset, SizedDatasetMapping


def train_test_split(): ...


def sequential_train_test_split[T](
    dataset: SizedDataset[T], train_ratio: float = 0.7
) -> tuple[SizedDatasetMapping[T], SizedDatasetMapping[T]]:
    num_points = len(dataset)
    num_train_points = int(train_ratio * num_points)

    train_dataset = SizedDatasetMapping(range(num_train_points), dataset)
    test_dataset = SizedDatasetMapping(range(num_train_points, num_points), dataset)

    return train_dataset, test_dataset


def cross_validation(): ...
