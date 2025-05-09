# Copyright (c) 2025 Yeelysia. All rights reserved.

import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from albumentations import Compose
from typing import Any,Optional,Union,List,Tuple
from pathlib import Path
from .augment import classify_augmentations, classify_transforms
from PIL import Image

class BaseDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, img_dir: str, label_dir: str, transform: Compose):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
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

    def load_image(self, img_path: str) -> Optional[np.ndarray[Any, Any]]:
        """Load an image from the specified path."""
        
        try:
            image = cv2.imread(img_path)
        except Exception as e:
            print(f"Warning: 无法读取图像 {img_path}, 错误: {str(e)}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_label(self, label_path: str) -> int:
        """Load a label from the specified path."""
        label = -1  # Default invalid label
        try:
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
                if label < 0 or label > 9:
                    print(f"Warning: 无效标签值 {label} 在文件 {label_path}")
                    label = -1
        except Exception as e:
            print(f"Warning: 无法读取或解析标签文件 {label_path},\n[ Error ] : {str(e)}")
        return label

class ClassifyDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Args:
        samples (list): 一个元组列表，每个元组包含图像的路径、其类别索引。
        torch_transforms (callable): 应用于图像的 PyTorch 变换。
        root (str): 数据集的根目录。

    Methods:
        __getitem__(i): 返回与给定索引对应的数据和目标子集。
        __len__(): 返回数据集中样本的总数。
    """

    def __init__(self, root: Union[str, Path], augment: bool = False):
        """
        使用根目录、增强初始化 ClassifyDataset 对象。

        Args:
            root (str): 数据集目录的路径。
            augment (bool, optional): 是否对数据集应用增强。
        """
        from torchvision.datasets import ImageFolder

        self.base = ImageFolder(root=root, allow_empty=True)

        self._samples : List[Tuple[str, int]] = self.base.samples # type: ignore
        self.root : Union[str, Path] = self.base.root # type: ignore
 

        self.samples : List[List[Any]] = list(x) for x in self._samples]  # file, index # type: ignore
 
        self.torch_transforms = classify_augmentations if augment else classify_transforms
        

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """
        返回与给定索引相对应的图像和类别索引。

        Args:
            i (int): 要检索的样本的索引。

        Returns:
            (dict): 包含图像及其类别索引的字典。
        """
        f, j= self.samples[i]  # filename, index
        im = cv2.imread(f)  # BGR

        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)['image']
        return sample, j

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)