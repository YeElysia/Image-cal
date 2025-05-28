# Copyright (c) 2025 Yeelysia. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Any, Optional

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体

def plot_training_history(history: dict[str, list[float]], title : str="Training History"):
    """
    绘制训练历史曲线

    参数:
        history: 包含训练历史的字典
        title: 图表标题
    """
    plt.figure(figsize=(12, 5)) # type: ignore

    # 绘制损失曲线
    plt.subplot(1, 2, 1) # type: ignore
    plt.plot(history['train_loss'], label='Training Loss') # type: ignore
    plt.plot(history['val_loss'], label='Validation Loss')  # type: ignore
    plt.xlabel('Epochs')  # type: ignore
    plt.ylabel('Loss')  # type: ignore
    plt.title('Loss Curves')  # type: ignore
    plt.legend() # type: ignore

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)  # type: ignore
    plt.plot(history['train_acc'], label='Training Accuracy')  # type: ignore 
    plt.plot(history['val_acc'], label='Validation Accuracy')  # type: ignore 
    plt.xlabel('Epochs')  # type: ignore
    plt.ylabel('Accuracy')  # type: ignore
    plt.title('Accuracy Curves') # type: ignore
    plt.legend() # type: ignore

    plt.suptitle(title)  # type: ignore
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png") # type: ignore
    plt.show() # type: ignore

def visualize_model_predictions(
        model : torch.nn.Module, 
        test_loader:torch.utils.data.DataLoader[tuple[torch.Tensor, Any]], 
        classes:dict[int, str], 
        device: Optional[torch.device] =None, 
        num_images:int=25
    ):
    """
    可视化模型预测

    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 使用的设备
        num_images: 要显示的图像数量
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # 获取batch数据
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)


    # 计算display_grid的尺寸
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15)) # type: ignore

    for i, ax in enumerate(axes.flat):
        if i < min(num_images, len(preds)):
            img = images[i].numpy().transpose((1, 2, 0))
            # 反标准化
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            color = "green" if preds[i] == labels[i] else "red"
            ax.set_title(f"Predicted: {classes[int(preds[i].item())]}\nTrue: {classes[int(labels[i].item())]}", color=color) 
        ax.axis('off')

    plt.tight_layout()
    # plt.suptitle("Model Predictions", fontsize=16) # type: ignore
    plt.savefig("model_predictions.png")  # type: ignore
    plt.show() # type: ignore


def plot_roc_curve(
    model : torch.nn.Module, 
    test_loader:torch.utils.data.DataLoader[tuple[torch.Tensor, Any]], 
    classes:dict[int, str], 
    device: Optional[torch.device] =None, 
    title: str = "Multiclass ROC Curve"
):
    """
    绘制多分类ROC曲线
    
    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        classes: 类别名称列表
        device: 使用的设备
        title: 图表标题
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    y_true = []
    y_score = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 将标签二值化
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 计算宏平均ROC曲线和AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘图设置
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    
    # 绘制每个类别的ROC曲线
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    # 绘制平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{title.replace(' ', '_')}.png")  # type: ignore
    plt.show()

