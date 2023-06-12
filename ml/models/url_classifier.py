import torch
from torch import nn


class URLClassifier(nn.Module):
    """
    URLClassifier is a PyTorch model class for malicious URL classification.
    
    Attributes:
    -----------
    embedding: torch.nn.Embedding
        An embedding layer that maps the URL characters to embeddings.
    pooling: torch.nn.AvgPool1d
        A pooling layer that reduces the dimensionality of the data.
    fc1: torch.nn.Linear
        The first fully connected layer.
    relu: torch.nn.ReLU
        A ReLU activation function.
    fc2: torch.nn.Linear
        The second fully connected layer.
    output: torch.nn.Sigmoid
        A Sigmoid activation function to produce final output.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 50, hidden_dim: int = 16, max_len: int = 100):
        """
        Instantiate the URLClassifier.
        
        Parameters:
        -----------
        vocab_size: int
            The size of the vocabulary (number of unique characters in the URLs).
        embed_dim: int, default 50
            The size of the character embeddings.
        hidden_dim: int, default 16
            The size of the hidden layer in the neural network.
        max_len: int, default 100
            The maximum length of the URLs (longer URLs will be truncated).
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(embed_dim // 2 * max_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the URLClassifier.
        
        Parameters:
        -----------
        x: torch.Tensor
            The input data (a batch of URLs).
            
        Returns:
        --------
        torch.Tensor
            The output of the classifier (predicted probabilities for the URLs being malicious).
        """
        x = self.embedding(x)
        x = self.pooling(x.transpose(1,2)).transpose(1,2)
        x = self.fc1(x.flatten(start_dim=1))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        return x
