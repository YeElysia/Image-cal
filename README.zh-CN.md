# 图片分类器

该项目使用 PyTorch 实现了用于对象检测的 YOLO（You Only Look Once）分类器。它包含数据加载、训练和评估功能，以及训练历史和模型预测的可视化。

## 目录

- [图片分类器](#图片分类器)
  - [目录](#目录)
  - [开源协议](#开源协议)
  - [安装](#安装)
  - [使用](#使用)
  - [Config 设置](#config-设置)

## 开源协议

- 本项目部分代码基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0 许可证）。
- 由于 AGPL 的限制，本项目整体采用 AGPL-3.0 许可证。
- 我第一次使用 AGPL-3.0 License，如有错误请指出

## 安装

1. 克隆该仓库:

   ```bash
   git clone https://github.com/YeElysia/Image-cal.git
   cd Image-cal
   ```

2. 安装依赖:
   本项目开发时使用 Python 版本为 3.10.16, pytorch 为"2.7.0+cu128"版本，如遇安装报错，请自行更换 pytorch 版本

   ```bash
   pip install -r requirements.txt
   ```

   确保你的 Pytorch 被安装。

## 使用

1. 按照以下结构准备数据集：

   ```
   dataset/
   ├── class_name_1/
   │   ├── *.jpg
   │   └── ...
   └── clss_name_2/
   │   ├── *.jpg
   │   └── ...
   └── ...
   ```

2. 更新 `main.py` 中的 `data_dir` 变量以指向您的数据集目录。
   你也可以更改`cfg/default.yaml` 中的 `dataset` 变量。

3. 运行训练脚本:
   ```bash
   python main.py
   ```

## Config 设置

配置文件 `default.yaml` 包含大部分的设置：

