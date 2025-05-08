# Image Classifier

This project implements a YOLO (You Only Look Once) classifier for object detection using PyTorch. It includes data loading, training, and evaluation functionalities, along with visualization of training history and model predictions.

[中文](https://github.com/YeElysia/Image-cal/blob/main/README.zh-CN.md)

## Table of Contents

- [Image Classifier](#image-classifier)
  - [Table of Contents](#table-of-contents)
  - [Open source licenses](#open-source-licenses)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Config setting](#config-setting)

## Open source licenses

- Part of the code of this project is based on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0 license).
- Due to the restrictions of AGPL, this project adopts the AGPL-3.0 license as a whole.
- This is my first time using AGPL-3.0 License. Please point out any errors.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YeElysia/Image-cal.git
   cd Image-cal
   ```

2. Install the required packages:
   This project was developed using Python version 3.10.16 and pytorch version "2.7.0+cu128". If you encounter an installation error, please change the pytorch version yourself.

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have PyTorch installed.

## Usage

1. Prepare your dataset in the following structure:

   ```
   dataset/
   ├── train/
   │   ├── images/
   │   │   ├── 00001.jpg
   │   │   └── ...
   │   └── labels/
   │   │   ├── 00001.txt
   │   │   └── ...
   └── val/
       ├── images/
       └── labels/
   ```

   The txt file only contains numbers representing categories. Please make sure to adjust the classes in the yaml file to match your categories.

2. Update the `data_dir` variable in `main.py` to point to your dataset directory.

3. Run the training script:
   ```bash
   python main.py
   ```

## Config setting

The configuration file `default.yaml` contains most of the settings:

