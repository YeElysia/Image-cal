import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from net.yolo import YOLO_CLA
from data.augment import get_transforms
from data.dataset import MyDataset
from utils.plotting import plot_training_history, visualize_model_predictions

class Trainer:
    def __init__(self, data_dir: str, num_classes: int, batch_size: int = 50, num_epochs: int = 20):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history: dict[str, list[float]] = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': [],
            'learning_rate': []
        }
        self.model = YOLO_CLA(num_classes=self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def load_data(self):
        train_img_dir = os.path.join(self.data_dir, 'train', 'images')
        train_label_dir = os.path.join(self.data_dir, 'train', 'labels')
        val_img_dir = os.path.join(self.data_dir, 'val', 'images')
        val_label_dir = os.path.join(self.data_dir, 'val', 'labels')

        # 数据增强和预处理
        train_transform, val_transform = get_transforms()

        # 创建数据集
        self.train_dataset = MyDataset(train_img_dir, train_label_dir, transform=train_transform)
        self.val_dataset = MyDataset(val_img_dir, val_label_dir, transform=val_transform)

        # 检查数据集
        if len(self.train_dataset) == 0:
            raise ValueError("训练数据集为空，请检查路径和文件")
        if len(self.val_dataset) == 0:
            raise ValueError("验证数据集为空，请检查路径和文件")

        print(f"\n数据集统计:")
        print(f"训练样本数: {len(self.train_dataset)}")
        print(f"验证样本数: {len(self.val_dataset)}")

        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def train(self):
        # 创建结果目录
        os.makedirs('test', exist_ok=True)
        best_acc = 0.0
        print("\n开始训练...")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            # 训练阶段
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 过滤无效样本
                valid_mask = (labels != -1) & (labels < self.num_classes)
                if not valid_mask.any():
                    continue
                    
                images = images[valid_mask]
                labels = labels[valid_mask]
                
                # 前向传播和反向传播
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()  # type: ignore
                
                # 统计信息
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 打印批次进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_loader)} | "
                          f"Loss: {loss.item():.4f} | Acc: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%")
            
            # 计算训练指标
            train_loss /= len(self.train_loader)
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            
            # 验证阶段
            val_loss, val_correct, val_total = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    valid_mask = (labels != -1) & (labels < self.num_classes)
                    if not valid_mask.any():
                        continue
                        
                    images = images[valid_mask]
                    labels = labels[valid_mask]
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(self.val_loader) if len(self.val_loader) > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # 更新学习率
            self.scheduler.step(val_acc)  # type: ignore
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(time.time() - epoch_start)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印epoch结果
            print(f"\nEpoch {epoch+1}/{self.num_epochs} | Time: {time.time() - epoch_start:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                }, 'test/best.pth')
                print(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        # 训练结束
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
        print(f"最佳验证准确率: {best_acc:.2f}%")
        
        # 可视化训练过程
        plot_training_history(self.history, title="Training History")
        
        # 可视化模型预测
        print("\n可视化验证集预测示例:")
        visualize_model_predictions(self.model, self.val_loader, classes, self.device)

if __name__ == "__main__":
    # 设置设备
    data_dir = '/mnt/E/dataset/'
    classes = ['RS', 'R1', 'R2', 'R3', 'R4', 'BS', 'B1', 'B2', 'B3', 'B4']
    num_classes = len(classes)

    trainer = Trainer(data_dir, num_classes)
    trainer.load_data()  # 加载数据
    trainer.train()  # 开始训练