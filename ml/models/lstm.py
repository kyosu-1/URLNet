import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """
    LSTM (Long Short-Term Memory) Classifier model class.

    This model uses an LSTM layer followed by a fully connected layer 
    and a sigmoid activation function for binary classification task.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary, i.e., maximum integer index + 1.
    embed_dim : int, optional
        The size of the vector space in which characters will be embedded, by default 50.
    hidden_dim : int, optional
        The number of features in the hidden state h of the LSTM layer, by default 256.
    num_layers : int, optional
        Number of recurrent layers (i.e., setting num_layers=2 would mean stacking two LSTMs together), by default 2.
    num_classes : int, optional
        The number of output classes, by default 1.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Defines the computation performed at every call.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 50, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        x = self.sigmoid(x)
        return x.squeeze()
