from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from ..core.distributions import LabeledSampleable


class MNISTSampler(nn.Module, LabeledSampleable):
    def __init__(
        self,
        root: str = "/kaggle/working/mnist",
        train: bool = True,
        img_size: int = 32,
        download: bool = True,
    ):
        super().__init__()
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1305,), (0.2891,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples random (image, label) pairs from the dataset.
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, 1, img_size, img_size)
            - labels: shape (batch_size,)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples ({num_samples}) exceeds dataset size ({len(self.dataset)})")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
