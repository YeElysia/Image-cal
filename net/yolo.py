import torch.nn as nn

class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 1)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            out += residual
        return out
    
class YOLO_CLABackbone(nn.Module):
    """YOLO-CLA主干网络"""
    def __init__(self):
        super().__init__()
        # 初始卷积层
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        
        # 下采样阶段
        self.down1 = nn.Sequential(
            ConvBlock(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64, 64)
        )
        
        self.down2 = nn.Sequential(
            ConvBlock(64, 128, 3, stride=2, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        
        self.down3 = nn.Sequential(
            ConvBlock(128, 256, 3, stride=2, padding=1),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        self.down4 = nn.Sequential(
            ConvBlock(256, 512, 3, stride=2, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )
        
        self.down5 = nn.Sequential(
            ConvBlock(512, 1024, 3, stride=2, padding=1),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x

class ClassificationHead(nn.Module):
    """分类头"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class YOLO_CLA(nn.Module):
    """完整的YOLO-CLA模型"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = YOLO_CLABackbone()
        self.head = ClassificationHead(1024, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    