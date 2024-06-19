import torch.nn as nn
import torch.nn.functional as F


class BinaryNet(nn.Module):
    def __init__(self,num_features):
        super(BinaryNet, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(num_features,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,2)

    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)