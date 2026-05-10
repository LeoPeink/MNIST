import torch
from DATA.ubyte_idx_reader import Mnistreader


class Dataset:

    def __init__(self, set_path=None, labels_path=None):
        self.data = Mnistreader(set_path).data
        self.labels = Mnistreader(labels_path).data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
