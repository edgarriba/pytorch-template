import torch.nn as nn


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        assert len(input.shape) == 4, input.shape
        assert input.shape == target.shape, (input.shape, target.shape)
        return self.loss(input, target)