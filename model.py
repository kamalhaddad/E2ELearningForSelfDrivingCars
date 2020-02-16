import torch.nn as nn

class Dave2Model(nn.Module):
    def __init__(self):
        super(Dave2Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(48, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1152, 1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1)
        )
    def forward():
        
