from src.data_manipulation.custom_dataset_abc import SizedDataset
from torch import Tensor
import torch


class SlidingWindowDataset(SizedDataset):
    def __init__(
        self, 
        sequence: Tensor,
        window_size: int, 
        step_size: int, 
        shift_size: int
    ) -> None:
        super().__init__()
        self.sequence = sequence
        self.window_size = window_size
        self.step_size = step_size
        self.shift_size = shift_size
        self.dataset_size = (sequence.shape[1] - window_size - shift_size + 1) // step_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        input_range = torch.arange(idx*self.step_size, idx*self.step_size + self.window_size)
        target_range = input_range + self.shift_size
        input_seq = self.sequence[:, input_range]
        target_seq = self.sequence[:, target_range]
        return (input_seq, target_seq)

    def __len__(self) -> int:
        return self.dataset_size
