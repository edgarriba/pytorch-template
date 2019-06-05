import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_outputs):
        super(MyModel, self).__init__()
        self.features = nn.Conv2d(3, num_outputs, 3, 1, 1)

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        return self.features(x)