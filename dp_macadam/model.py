import torch.nn as nn


class AdaClipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def make_fresh_model(device):
    model = AdaClipNet().to(device)
    model.train()
    return model

