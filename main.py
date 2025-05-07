import torch
import torch.nn as nn
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
    
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

from net.yolo import YOLO_CLA
from albumentations import Compose

from data.augment import get_transforms




class MyDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, img_dir:str, label_dir:str, transform: Compose):
        """
        Args:
            img_dir (string): 图片目录路径
            label_dir (string): 标签目录路径
            transform (callable, optional): 可选的数据变换
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 获取所有图片文件
        self.img_files = [f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
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
    
    def __getitem__(self, idx):
        img_file = self.valid_samples[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        # 初始化默认值
        image = Image.new('RGB', (224, 224), (0, 0, 0))  # 保持为PIL Image
        label = -1  # 使用无效标签
        is_valid = True  # 标记样本是否有效
        
        # 读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: 无法加载图片 {img_path}, 错误: {str(e)}")
            is_valid = False
        
        # 读取标签
        try:
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
                if label < 0 or label > 9:
                    print(f"Warning: 无效标签值 {label} 在文件 {label_path}")
                    label = -1
                    is_valid = False
        except Exception as e:
            print(f"Warning: 无法读取或解析标签文件 {label_path}, 错误: {str(e)}")
            is_valid = False
        
        # 应用transform
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Warning: 无法转换图像 {img_path}, 错误: {str(e)}")
                # 使用默认转换
                image = transforms.ToTensor()(Image.new('RGB', (224, 224), (0, 0, 0)))
                is_valid = False
        
        return image, label
def plot_training_history(history, title="Training History"):
    """
    绘制训练历史曲线

    参数:
        history: 包含训练历史的字典
        title: 图表标题
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')  # 英文标签
    plt.plot(history['val_loss'], label='Validation Loss')  # 英文标签
    plt.xlabel('Epochs')  # 英文标签
    plt.ylabel('Loss')  # 英文标签
    plt.title('Loss Curves')  # 英文标题
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')  # 英文标签
    plt.plot(history['val_acc'], label='Validation Accuracy')  # 英文标签
    plt.xlabel('Epochs')  # 英文标签
    plt.ylabel('Accuracy')  # 英文标签
    plt.title('Accuracy Curves')  # 英文标题
    plt.legend()

    plt.suptitle(title)  # 英文总标题
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def visualize_model_predictions(model, test_loader, classes, device=None, num_images=25):
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

    # 将预测和标签转换为CPU上的numpy数组
    preds = preds.cpu().numpy()
    labels = labels.numpy()

    # 计算display_grid的尺寸
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

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
            ax.set_title(f"Predicted: {classes[preds[i]]}\nTrue: {classes[labels[i]]}", color=color)  # 英文标签
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def train():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    start_time = time.time()

    # 创建结果目录
    os.makedirs('test', exist_ok=True)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': [],
        'learning_rate': []
    }
    
    # 数据路径
    data_dir = '/mnt/E/dataset/'
    train_img_dir = os.path.join(data_dir, 'train', 'images')
    train_label_dir = os.path.join(data_dir, 'train', 'labels')
    val_img_dir = os.path.join(data_dir, 'val', 'images')
    val_label_dir = os.path.join(data_dir, 'val', 'labels')
    
    classes = ('RS','R1','R2','R3','R4','BS','B1','B2','B3','B4')
    num_classes = len(classes)
    
    # 数据增强和预处理
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    train_dataset = MyDataset(train_img_dir, train_label_dir, transform=train_transform)
    val_dataset = MyDataset(val_img_dir, val_label_dir, transform=val_transform)
    
    # 检查数据集
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空，请检查路径和文件")
    if len(val_dataset) == 0:
        raise ValueError("验证数据集为空，请检查路径和文件")
    
    print(f"\n数据集统计:")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 数据加载器
    batch_size = 50
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True)
    
    model = YOLO_CLA(num_classes=num_classes)
    # checkpoint = torch.load('test/best.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

     # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1)
    
    # 训练参数
    num_epochs = 20
    best_acc = 0.0
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # 训练阶段
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 过滤无效样本
            valid_mask = (labels != -1) & (labels < num_classes)
            if not valid_mask.any():
                continue
                
            images = images[valid_mask]
            labels = labels[valid_mask]
            
            # 前向传播和反向传播
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计信息
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 打印批次进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%")
        
        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # 验证阶段
        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                valid_mask = (labels != -1) & (labels < num_classes)
                if not valid_mask.any():
                    continue
                    
                images = images[valid_mask]
                labels = labels[valid_mask]
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(time.time() - epoch_start)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{num_epochs} | Time: {time.time() - epoch_start:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
            }, 'test/best.pth')
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
    
    # 训练结束
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    
    # 可视化训练过程
    plot_training_history(history, title="Training History")
    
    # 可视化模型预测
    print("\n可视化验证集预测示例:")
    visualize_model_predictions(model, val_loader, classes, device)
    
    return model, history

if __name__ == "__main__":
    model, history = train()