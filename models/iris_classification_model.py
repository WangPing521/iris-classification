import torch.nn as nn
from torch.nn import functional as F


class iris_classifier(nn.Module):
    def __init__(self, num_class=3):
        super(iris_classifier, self).__init__()
        self.conv1 = nn.Linear(4, 10)
        self.conv2 = nn.Linear(10, 20)
        self.conv3 = nn.Linear(20, 10)
        self.conv4 = nn.Linear(10, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        return out
