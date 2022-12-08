from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        output = self.model(x)
        return F.log_softmax(output, dim=-1)
