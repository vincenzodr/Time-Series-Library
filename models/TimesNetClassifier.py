import torch
import torch.nn as nn
import torch.nn.functional as F
import TimesNet

class Model(nn.Module):

    def __init__(self, configs, h1 = 8, h2 = 9, num_classes = 16):
        super(Model, self).__init__()
        configs.task_name = 'anomaly_detection'
        self.timesnet = TimesNet.Model(configs)
        self.fc1 = nn.Linear(configs.c_out, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, num_classes)

    def forward(self, x):
        x = self.timesnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

