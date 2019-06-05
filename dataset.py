import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_root=None, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.data_index = self.build_index(self.data_root)

    def build_index(self, data_root):
        return [None, None]

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        sample = self.data_index[idx]

        # load data, NOTE: modify by cv2.imread(...)
        image = torch.rand(3, 240, 320)
        label = torch.rand(1, 240, 320)
        return dict(images=image, labels=label)