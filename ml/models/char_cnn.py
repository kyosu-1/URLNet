import torch
from torch import nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 50, num_filters: int = 256,
                 filter_sizes: list[int] = [7, 7, 3, 3, 3, 3], num_classes: int = 1):
        """
        Character-level Convolutional Neural Network (CharCNN) for detecting malicious URLs.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        embed_dim : int, optional
            The dimensionality of the character embeddings. Defaults to 50.
        num_filters : int, optional
            The number of filters in each convolutional layer. Defaults to 256.
        filter_sizes : list[int], optional
            The kernel sizes for the convolutional layers. Defaults to [7, 7, 3, 3, 3, 3].
        num_classes : int, optional
            The number of output classes. Defaults to 1.
        """
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
                                    for fs in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the CharCNN.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, sequence_length).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size).
        """
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()
