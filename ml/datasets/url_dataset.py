import torch
from torch.utils.data import Dataset
from collections import Counter


class URLDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int], max_len: int = 100, min_freq: int = 2, char2idx: dict[str, int] = None):
        """
        Initialize the dataset.

        Args:
        urls (list[str]): List of URL strings.
        labels (list[int]): List of labels (0 or 1).
        max_len (int): Maximum length of the URL strings.
        min_freq (int): Minimum frequency for a character to be included in the vocabulary.
        char2idx (dict[str, int], optional): Mapping from characters to indices.
        """
        self.urls = urls
        self.labels = labels
        self.max_len = max_len

        if char2idx is None:
            counter = Counter(''.join(urls))
            self.char2idx = {char: idx for idx, (char, freq) in enumerate(counter.items()) if freq >= min_freq}
        else:
            self.char2idx = char2idx

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.urls)

    def __getitem__(self, idx):
        """
        Returns the item at index 'idx'.

        Args:
        idx (int): Index.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: URL and label at index 'idx'.
        """
        url = self.urls[idx]
        label = self.labels[idx]
        
        url_tensor = torch.zeros(self.max_len).long()
        for i, char in enumerate(url[:self.max_len]):
            if char in self.char2idx:
                url_tensor[i] = self.char2idx[char]

        return url_tensor, torch.tensor(label)
