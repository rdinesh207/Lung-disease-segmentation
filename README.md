# Lung-disease-segmentation-VinDr-CXR
VINDR-CXR Chest X-ray Abnormality Detection using Swin Transformer and Faster RCNN

Overview

This project implements a Faster RCNN object detection model with a Swin Transformer backbone to detect abnormalities in chest X-ray images from the VINDR-CXR dataset. The model is trained to identify and localize 23 different classes of findings, including "No finding" and various lung pathologies.

Dataset

This project utilizes the VINDR-CXR dataset, a large-scale chest X-ray dataset for abnormality detection. The dataset is publicly available on Kaggle:

VINDR-CXR Dataset on Kaggle: https://www.kaggle.com/datasets/vinmecdata/vindrcxr

The VINDR-CXR dataset contains chest X-ray images in DICOM format and bounding box annotations for various thoracic abnormalities. This code expects the dataset to be structured with train and test folders containing JPEG images and corresponding annotation CSV files.

Model Architecture

The object detection model is based on Faster RCNN with a Swin Transformer as the backbone.

* Faster RCNN: A popular object detection framework that consists of two stages:
    * Region Proposal Network (RPN): Proposes candidate object bounding boxes.
    * Fast R-CNN: Extracts features from the proposed boxes and performs classification and bounding box regression.
* Swin Transformer: A hierarchical Vision Transformer that serves as a powerful feature extractor. It introduces the shifted window attention mechanism, which brings greater efficiency and flexibility compared to previous Vision Transformers.

Reference Paper:

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows: https://arxiv.org/abs/2103.14030

This implementation leverages pre-trained weights for the Swin Transformer backbone from `torchvision.models`.

Classes

The model is trained to detect the following 23 classes:

classes = {
    "No finding": 0,
    "Aortic enlargement": 1,
    "Atelectasis": 2,
    "Calcification": 3,
    "Cardiomegaly": 4,
    "Clavicle fracture": 5,
    "Consolidation": 6,
    "Edema": 7,
    "Emphysema": 8,
    "Enlarged PA": 9,
    "ILD": 10,
    "Infiltration": 11,
    "Lung cavity": 12,
    "Lung cyst": 13,
    "Lung Opacity": 14,
    "Mediastinal shift": 15,
    "Nodule/Mass": 16,
    "Other lesion": 17,
    "Pleural effusion": 18,
    "Pleural thickening": 19,
    "Pneumothorax": 20,
    "Pulmonary fibrosis": 21,
    "Rib fracture": 22
}

Prerequisites

* Python 3.x
* PyTorch
* Torchvision
* PIL (Pillow)
* Matplotlib
* OpenCV
* Pandas
* Scikit-learn
* IPython

You can install the required packages using pip:

pip install torch torchvision torchaudio matplotlib opencv-python pandas scikit-learn ipython Pillow

Data Preparation

1. Download the VINDR-CXR dataset from the Kaggle link provided above.

2. Organize the dataset:  The code expects the following directory structure:

    vindr-cxr/
    ├── train/
    │   ├── <image1>.dicom
    │   ├── <image2>.dicom
    │   └── ...
    ├── test/
    │   ├── <image1>.dicom
    │   ├── <image2>.dicom
    │   └── ...
    └── annotations/
        ├── annotations_train.csv
        ├── annotations_train_unique.csv
        └── annotations_test.csv

3. Convert DICOM images to JPEG: The provided code expects JPEG images. You will need to convert the DICOM images from the downloaded dataset to JPEG format and place them in `vindr_jpegs/train/` and `vindr_jpegs/test/` directories. You can adapt the provided code or use separate scripts for DICOM to JPEG conversion. Ensure the image filenames in the annotation CSV files match the JPEG filenames (without the `.dicom` extension, using `.jpeg` instead).

Usage

1. Clone this repository.

2. Prepare the dataset as described in the "Data Preparation" section.

3. Run the `swin_fasterrcnn.py` script:

    python swin_fasterrcnn.py

    This script will:

    * Create custom datasets for training and testing.
    * Initialize a Faster RCNN model with a Swin Transformer backbone.
    * Train the model for a specified number of epochs (default is 5).
    * Evaluate the model on the test dataset after each epoch.
    * Save the model weights after the training process.

Code Structure

* `swin_fasterrcnn.py`: Contains the main training and evaluation loop, dataset loading, model definition, and training configurations.
* `engine.py`, `utils.py`, `coco_utils.py`, `coco_eval.py`, `transforms.py`: These files are directly taken from the PyTorch Vision references for object detection and provide utility functions for training, evaluation, and data handling. They are essential for running the training script.

Training Details

* Epochs: 5 (configurable in `swin_fasterrcnn.py`)
* Batch Size: 6 (configurable in `swin_fasterrcnn.py`)
* Optimizer: Adam
* Learning Rate: 0.001
* Learning Rate Scheduler: StepLR (steps every 3 epochs, gamma=0.1)
* Device: GPU if available, otherwise CPU

Model Saving

The trained model weights are saved after each epoch in `.pth` files named with the following format: `swin_fastercnn_model_YYMMDD_HHMMSS.pth`.

Future Work

* Experiment with different backbones (e.g., ResNet, other Vision Transformers).
* Fine-tune hyperparameters for improved performance.
* Implement data augmentation techniques.
* Evaluate the model using more comprehensive metrics (e.g., mAP).
* Develop an inference script for deploying the trained model.

License

MIT License: https://www.google.com/url?sa=E&source=gmail&q=LICENSE
