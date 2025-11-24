# models/mnist_cnn.py

import torch.nn as nn

class CentralizedMNISTCNN(nn.Module):
    def __init__(self):
        super(CentralizedMNISTCNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),  # 10 digit classes (0–9)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
