# Copyright (c) 2025 Yeelysia. All rights reserved.

import albumentations as A
from albumentations.pytorch import ToTensorV2


classify_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5), # 随机水平翻转
            A.Resize(height=224, width=224),
            A.RandomCrop(height=224, width=224, p=0.5), # 随机裁剪
            A.RandomBrightnessContrast(p=0.2), # 随机亮度对比度调整
            A.HueSaturationValue(p=0.2), # 随机色调饱和度值调整
            A.MotionBlur(blur_limit=3, p=0.2), # 随机运动模糊
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

classify_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])