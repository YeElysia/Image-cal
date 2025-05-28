# Copyright (c) 2025 Yeelysia. All rights reserved.

from logging import critical
import torch
import torch.nn as nn\

import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

import time
from net.yolo import YOLO_CLA
from net.resnet import ResNet, ResBlock
from net.mobilenet import MobileNetV2

from data.dataset import split_dataset
from utils.plotting import plot_training_history, visualize_model_predictions,plot_roc_curve
from utils.cfg import get_cfg
from typing import Any
from utils.Logger import logger
from utils import model_evaluate    

class Trainer:
    def __init__(self, **kwargs: Any):
        self.args = get_cfg(overrides=kwargs)
        self.dataset_path = self.args.dataset
        self.num_classes = self.args.num_classes
        self.batch_size = self.args.batch_size
        self.num_epochs = self.args.epochs

        self.device = torch.device(self.args.device)

        if self.device.type == 'cuda' and torch.cuda.is_available():
            logger.info(f"使用GPU进行训练")
            pass
        else:
            logger.warning(f"GPU不可用，请检查当前pytorch版本是否支持GPU")
            logger.info(f"使用CPU进行训练")
            self.device = torch.device('cpu')


        self.history: dict[str, list[float]] = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': [],
            'learning_rate': []
        }
        model_list = [YOLO_CLA(num_classes=self.num_classes), ResNet(ResBlock, [2, 2, 2, 2],num_classes=self.num_classes), MobileNetV2(num_classes=self.num_classes)]
        self.model = model_list[self.args.model].to(self.device)
        
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        self.data_load = False

        self.visible = self.args.visible

    def get_criterion(self):
        if self.args.criterion == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif self.args.criterion == 'FocalLoss':
            return nn.FocalLoss()
        else:
            raise ValueError(f"不支持的损失函数: {self.args.criterion}")
        
    def get_optimizer(self):
        if self.args.optimizer == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {self.args.optimizer}")
    
    def get_scheduler(self):
        if self.args.scheduler == 'StepLR':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.scheduler == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        else:
            raise ValueError(f"不支持的调度器: {self.args.scheduler}")
        

    def load_data(self):
        # 创建数据集
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(self.dataset_path)

        # 检查数据集
        if len(self.train_dataset) == 0:
            logger.error("训练数据集为空，请检查路径和文件")
            raise ValueError("训练数据集为空，请检查路径和文件")
        if len(self.val_dataset) == 0:
            logger.error("验证数据集为空，请检查路径和文件")
            raise ValueError("验证数据集为空，请检查路径和文件")

        logger.info(f"数据集统计:")
        logger.info(f"训练样本数: {len(self.train_dataset)}")
        logger.info(f"验证样本数: {len(self.val_dataset)}")

        # 数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.data_load = True

    def train(self):
        if not self.data_load:
            logger.warning("数据未加载，请先调用load_data方法")
            self.load_data()

        # 创建结果目录
        os.makedirs('test', exist_ok=True)
        best_acc = 0.0
        logger.info("开始训练...")
        start_time = time.time()
        
        train_loss, train_correct, train_total = 0.0, 0, 0
        loss = 0.0
        for epoch in range(self.num_epochs):
                        
            epoch_start = time.time()
            self.model.train()

            
            # 使用 tqdm 显示训练进度条
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            pbar.write(("%11s" * 2)
                        % (
                            "Epoch",
                            "Loss"
                        ))
            for _, (images, labels) in enumerate(pbar):
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
                
                loss = loss.item()
                
                # 动态更新：loss 和 acc
                pbar.set_description(                        
                    ("%11s" * 2)
                        % (
                            f"{epoch + 1}/{self.num_epochs}",
                            f"{loss:.4f}",
                        )
                    )

            
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
            self.scheduler.step()  # type: ignore
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(time.time() - epoch_start)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印epoch结果
            logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
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
                logger.info(f"Saved best model with val_acc: {val_acc:.2f}% at epoch {epoch+1}")
        
        # 训练结束
        total_time = time.time() - start_time
        logger.info(f"训练完成! 总时间: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
        logger.info(f"最佳验证准确率: {best_acc:.2f}%")

        # 可视化训练过程
        if self.visible:
            plot_training_history(self.history, title="Training History")
            visualize_model_predictions(self.model, self.val_loader, classes, self.device)
            plot_roc_curve(self.model, self.val_loader, classes, self.device)
            model_evaluate(self.model, self.val_loader, classes, self.device)


if __name__ == "__main__":
    # 设置设备
    data_dir = '/home/yee/.fiz/courseware/人工智能基础（A）/中药数据集'
    classes =  {0: '僵蚕', 1: '党参', 2: '天南星', 3: '枸杞', 4: '槐花', 5: '牛蒡子', 6: '牡丹皮', 7: '猪苓', 8: '甘草', 9: '百合', 10: '百部', 11: '竹叶', 12: '竹茹', 13: '紫草', 14: '红藤', 15: '艾叶', 16: '荆芥', 17: '金银花', 18: '黄柏', 19: '黄芪'}
    num_classes = len(classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    trainer = Trainer(model=1,dataset=data_dir, num_classes=num_classes, batch_size=10, epochs=10, device=device)
    trainer.load_data()
    trainer.train()  # 开始训练