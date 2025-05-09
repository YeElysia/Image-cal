# Copyright (c) 2025 Yeelysia. All rights reserved.

import numpy as np

import torch
from torch.utils.data import Dataset,Subset

from typing import Any,List,Tuple
from .augment import classify_augmentations, classify_transforms
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

    def __init__(self, samples, augment: bool = False):
        """
        使用根目录、增强初始化 ClassifyDataset 对象。

        Args:
            root (str): 数据集目录的路径。
            augment (bool, optional): 是否对数据集应用增强。
        """

        self.samples : List[List[Any]] = [ list(x) for x in samples]  # file, index # type: ignore
 
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

        # Convert NumPy array to PIL image
        im = np.array(f)
        sample = self.torch_transforms(image=im)['image']
        return sample, j

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    

def split_dataset(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[ClassifyDataset, ClassifyDataset, ClassifyDataset]:
    """
    将数据集划分为训练集、验证集和测试集。

    Args:
        data_path (str): 数据集的根目录路径。
        train_ratio (float): 训练集的比例。
        val_ratio (float): 验证集的比例。

    Returns:
        Tuple[ClassifyDataset, ClassifyDataset, ClassifyDataset]: 包含训练集、验证集和测试集的元组。
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import random_split

    # 获取所有样本
    dataset = ImageFolder(root=data_path)
    total_samples = len(dataset)

    # 计算训练集、验证集和测试集的数量
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples

    # 随机划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_samples, val_samples, test_samples])

    # 将划分后的数据集转换为 ClassifyDataset 实例
    train_classify_dataset = ClassifyDataset(train_dataset)  # 这里需要根据实际情况调整
    val_classify_dataset = ClassifyDataset(val_dataset)    # 这里需要根据实际情况调整
    test_classify_dataset = ClassifyDataset(test_dataset)   # 这里需要根据实际情况调整

    print(train_classify_dataset.samples[0])

    return train_classify_dataset, val_classify_dataset, test_classify_dataset



