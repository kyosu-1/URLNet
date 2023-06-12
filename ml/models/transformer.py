from torch import nn


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for detecting malicious URLs.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    embed_dim : int, optional
        The dimension of the embedding. Defaults to 50.
    num_heads : int, optional
        The number of attention heads in the transformer. Defaults to 2.
    hidden_dim : int, optional
        The dimension of the hidden layer in the transformer. Defaults to 256.
    num_layers : int, optional
        The number of transformer layers. Defaults to 2.
    num_classes : int, optional
        The number of output classes. Defaults to 1.

    """

    def __init__(self, vocab_size, embed_dim=50, num_heads=2, hidden_dim=256, num_layers=2, num_classes=1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim), num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the TransformerClassifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, num_classes).

        """
        x = self.embedding(x)
        x = self.transformer(x.transpose(0, 1))
        x = self.fc(x.mean(axis=0))
        x = self.sigmoid(x)
        return x.squeeze()