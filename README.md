# YOLO Classifier Training

This project implements a YOLO (You Only Look Once) classifier for object detection using PyTorch. It includes data loading, training, and evaluation functionalities, along with visualization of training history and model predictions.

## Table of Contents

- [YOLO Classifier Training](#yolo-classifier-training)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Training](#training)
  - [Visualization](#visualization)
  - [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have PyTorch installed. You can find installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

1. Prepare your dataset in the following structure:

   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/
   ```

2. Update the `data_dir` variable in `main.py` to point to your dataset directory.

3. Run the training script:
   ```bash
   python main.py
   ```

## Training

The `Trainer` class encapsulates the training logic. You can customize the following parameters in the `__init__` method:

- `data_dir`: Path to the dataset.
- `num_classes`: Number of classes in your dataset.
- `batch_size`: Number of samples per batch (default is 50).
- `num_epochs`: Number of training epochs (default is 20).

## Visualization

After training, the model's training history and predictions on the validation set will be visualized. The `plot_training_history` and `visualize_model_predictions` functions handle this.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
