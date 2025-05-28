import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1, 1, 0))

        layers.append(ConvBNReLU(hidden_dim, hidden_dim, 3, stride, 1))
        layers.append(ConvBNReLU(hidden_dim, out_channels, 1, 1, 0))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.stage_repeats = [1, 2, 3, 4]
        self.in_channels = 32
        self.last_channel = 1280

        self.features = [ConvBNReLU(3, 32, 3, 2, 1)]

        for stage_repeat in self.stage_repeats:
            output_channel = self.in_channels
            for _ in range(stage_repeat):
                self.features.append(InvertedResidual(self.in_channels, self.last_channel, 1, 6))
                self.in_channels = self.last_channel
            self.last_channel = self.in_channels

        self.features.append(ConvBNReLU(self.in_channels, 1280, 1, 1, 0))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )
  
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
