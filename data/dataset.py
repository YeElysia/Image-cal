# Copyright (c) 2025 Yeelysia. All rights reserved.
import cv2

from typing import Any,Union,List,Tuple
from pathlib import Path

from torchvision.datasets import ImageFolder
from PIL import Image

from .augment import classify_augmentations, classify_transforms

class ClassifyDataset:
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


        self.base = ImageFolder(root=root, allow_empty=True)

        self._samples : List[Tuple[str, int]] = self.base.samples # type: ignore
        self.root : Union[str, Path] = self.base.root # type: ignore
 

        self.samples : List[List[Any]] = list(x) for x in self._samples]  # file, index # type: ignore
 
        self.torch_transforms = classify_augmentations if augment else classify_transforms
        

    def __getitem__(self, i: int) -> dict[str, Any]:
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
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)