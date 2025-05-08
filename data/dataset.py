# Copyright (c) 2025 Yeelysia. All rights reserved.

import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from albumentations import Compose

class MyDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, img_dir:str, label_dir:str, transform: Compose):
        """
        Args:
            img_dir (string): 图片目录路径
            label_dir (string): 标签目录路径
            transform (callable, optional): 可选的数据变换

        Returns:
            tuple: (image, label) 其中image是PIL图像，label是整数
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform= transform
        
        # 获取所有图片文件
        self.img_files = [f for f in os.listdir(img_dir)]
        
        # 验证每个图片是否有对应的标签文件
        self.valid_samples: list[str] = []
        for img_file in self.img_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.valid_samples.append(img_file)
            else:
                print(f"Warning: 缺少标签文件 {label_file}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx:int): 
        img_file = self.valid_samples[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        # 初始化默认值
        image = np.zeros((256, 256, 3), dtype=np.uint8)  # 默认图像
        label = -1  # 使用无效标签
        
        # 读取图片
        try:
            image = cv2.imread(img_path)
            # RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: 无法加载图片 {img_path},\n[ Error ]: {str(e)}")
        
        # 读取标签
        try:
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
                if label < 0 or label > 9:
                    print(f"Warning: 无效标签值 {label} 在文件 {label_path}")
                    label = -1
        except Exception as e:
            print(f"Warning: 无法读取或解析标签文件 {label_path},\n[ Error ] : {str(e)}")
        

        try:
            img_tensor: torch.Tensor = self.transform(image=image)["image"]
        except Exception as e:
            print(f"Warning: 无法转换图像 {img_path}, 错误: {str(e)}")
            img_tensor = torch.zeros((3, 256, 256), dtype=torch.uint8)
        
        return img_tensor, label