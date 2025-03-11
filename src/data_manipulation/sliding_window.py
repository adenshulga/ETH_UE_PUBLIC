from src.data_manipulation.custom_dataset_abc import SizedDataset
from torch import Tensor
import torch


class SlidingWindowDataset(SizedDataset[tuple[Tensor, Tensor]]):
    """
    Sliding window dataset iterates over pairs of an input sequnce and an output 
    sequence, where the output sequnce is shifted from the input sequcen by a 
    shift size.
    """
    def __init__(
        self, 
        sequence: SizedDataset[Tensor],
        window_size: int, 
        step_size: int, 
        shift_size: int
    ) -> None:
        """
        Args:
            sequence: the sequence of the shape (d, l) where d is the
                dimensionality, l is the sequence length.
            window_size: the size of the window to slide over the sequence.
            step_size: the step size to slide over the sequence.
            shift_size: the shift size between input and output sequences.
        """
        super().__init__()
        self.sequence = torch.tensor(sequence)
        self.window_size = window_size
        self.step_size = step_size
        self.shift_size = shift_size
        self.dataset_size = (
            self.sequence.shape[1] - window_size - shift_size + 1) // step_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            idx: the index of the requested element.
        Returns:
            tuple[Tensor, Tensor]: the pair of input and output sequences. Each 
                sequence of the shape (d, l) where d is the dimensionality, l
                is the window size.
        """
        input_range = torch.arange(
            idx*self.step_size, idx*self.step_size + self.window_size)
        target_range = input_range + self.shift_size
        input_seq = self.sequence[:, input_range]
        target_seq = self.sequence[:, target_range]
        return (input_seq, target_seq)

    def __len__(self) -> int:
        return self.dataset_size
