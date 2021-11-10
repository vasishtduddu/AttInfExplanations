import torch.nn as nn
import torch.nn.functional as F


class BinaryNet(nn.Module):
    def __init__(self,num_features):
        super(BinaryNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(num_features, 1024),nn.Tanh(),)
        self.fc2 = nn.Sequential(nn.Linear(1024, 512),nn.Tanh(),)
        self.fc3 = nn.Sequential(nn.Linear(512, 256),nn.Tanh(),)
        self.fc4 = nn.Sequential(nn.Linear(256, 128),nn.Tanh(),)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)
