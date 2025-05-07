import matplotlib.pyplot as plt
import numpy as np
import torch

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

def visualize_model_predictions(model : torch.nn.Module, test_loader:torch.utils.data.DataLoader[tuple[torch.Tensor, int]], classes:list[str], device: torch.device | None =None, num_images:int=25):
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
            ax.set_title(f"Predicted: {classes[preds[i]]}\nTrue: {classes[labels[i]]}", color=color) 
        ax.axis('off')

    plt.tight_layout()
    plt.show() # type: ignore
