from torch import nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        self.fc = nn.Linear(10 * 28 * 28, 10)

    def forward(self, x):
        x = self.cnn1(x)
        output = self.cnn2(x)
        x = x + output
        output = self.cnn2(x)
        x = x + output
        x = self.cnn3(x)
        output = self.cnn4(x)
        x = x + output
        output = self.cnn4(x)
        x = x + output
        x = self.cnn5(x)
        x = x.view(-1, 10 * 28 * 28)
        output = self.fc(x)
        return F.log_softmax(output, dim=-1)
