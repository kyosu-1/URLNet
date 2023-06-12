import torch
import torch.nn.functional as F
from torch import nn


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text classification implemented in PyTorch.
    This network is designed to take three types of encoded URL as input, perform convolution
    and max pooling over each, and then combine the results for final classification.
    """

    def __init__(
        self,
        char_ngram_vocab_size: int,
        word_ngram_vocab_size: int,
        char_vocab_size: int,
        word_seq_len: int,
        char_seq_len: int,
        embedding_size: int,
        mode: int = 0,
        filter_sizes: list[int] = [3, 4, 5, 6],
        l2_reg_lambda: float = 0,
    ):
        """
        Initialize the TextCNN model with the provided parameters.

        Args:
            char_ngram_vocab_size (int): The number of unique character n-grams in the dataset.
            word_ngram_vocab_size (int): The number of unique word n-grams in the dataset.
            char_vocab_size (int): The number of unique characters in the dataset.
            word_seq_len (int): The length of the word sequences.
            char_seq_len (int): The length of the character sequences.
            embedding_size (int): The size of the embedding vectors.
            mode (int, optional): The mode of operation of the network. Defaults to 0.
            filter_sizes (List[int], optional): The sizes of the filters to use in the convolutional layers. Defaults to [3,4,5,6].
            l2_reg_lambda (float, optional): The L2 regularization strength. Defaults to 0.
        """
        super(TextCNN, self).__init__()

        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        self.mode = mode

        if mode in [4, 5]:
            self.char_emb = nn.Embedding(char_ngram_vocab_size, embedding_size, padding_idx=0)
        if mode in [2, 3, 4, 5]:
            self.word_emb = nn.Embedding(word_ngram_vocab_size, embedding_size, padding_idx=0)
        if mode in [1, 3, 5]:
            self.char_seq_emb = nn.Embedding(char_vocab_size, embedding_size, padding_idx=0)

        if mode in [2, 3, 4, 5]:
            self.conv_layers_word = nn.ModuleList(
                [nn.Conv2d(1, 256, (fsize, embedding_size), padding=(fsize - 1, 0)) for fsize in filter_sizes]
            )

        if mode in [1, 3, 5]:
            self.conv_layers_char = nn.ModuleList(
                [nn.Conv2d(1, 256, (fsize, embedding_size), padding=(fsize - 1, 0)) for fsize in filter_sizes]
            )

        self.fc1 = nn.Linear(256 * len(filter_sizes), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(p=1 - l2_reg_lambda)

    def forward(self, x_word: torch.Tensor, x_char: torch.Tensor, x_char_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TextCNN network.

        Args:
            x_word (torch.Tensor): The tensor of word n-grams. Shape should be (batch_size, seq_length).
            x_char (torch.Tensor): The tensor of character n-grams. Shape should be (batch_size, seq_length, ngram_length).
            x_char_seq (torch.Tensor): The tensor of character sequences. Shape should be (batch_size, seq_length).

        Returns:
            torch.Tensor: The output probabilities of the network. Shape is (batch_size, num_classes).
        """
        if self.mode in [4, 5]:
            x_char = self.char_emb(x_char)
        if self.mode in [2, 3, 4, 5]:
            x_word = self.word_emb(x_word)
        if self.mode in [1, 3, 5]:
            x_char_seq = self.char_seq_emb(x_char_seq)

        if self.mode in [4, 5]:
            x_char = x_char.sum(2)
            x_word = x_char + x_word

        if self.mode in [2, 3, 4, 5]:
            x_word = x_word.unsqueeze(1)
            x_word = [F.relu(conv(x_word)).squeeze(3) for conv in self.conv_layers_word]
            x_word = [F.max_pool1d(x, x.size(2)) for x in x_word]
            x_word = torch.cat(x_word, dim=1).squeeze(2)
            x_word = self.dropout(x_word)

        if self.mode in [1, 3, 5]:
            x_char_seq = x_char_seq.unsqueeze(1)
            x_char_seq = [F.relu(conv(x_char_seq)).squeeze(3) for conv in self.conv_layers_char]
            x_char_seq = [F.max_pool1d(x, x.size(2)) for x in x_char_seq]
            x_char_seq = torch.cat(x_char_seq, dim=1).squeeze(2)
            x_char_seq = self.dropout(x_char_seq)

        if self.mode in [3, 5]:
            conv_output = torch.cat([x_word, x_char_seq], 1)
        elif self.mode in [2, 4]:
            conv_output = x_word
        elif self.mode == 1:
            conv_output = x_char_seq

        output = F.relu(self.fc1(conv_output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        scores = self.fc4(output)
        probs = F.softmax(scores, dim=1)

        return probs
