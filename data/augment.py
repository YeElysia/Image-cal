import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms


# 定义训练和验证的转换
train_transform = A.Compose([
    A.Resize(height = 224 , width = 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def get_transforms():
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=28), # 随机裁剪
        # Cutout(p=0.5), # 使用Cutout正则化
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform