from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import sys
import copy
import torch
import time
import pydicom
import numpy as np
import IProgress
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from IPython.display import display, clear_output
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.models.detection import MaskRCNN, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def main():
    
    # Define train and test folders
    train_data ="physionet.org/files/vindr-cxr/1.0.0/train"
    train_annotations ="physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train.csv"
    train_annotations_unique ="physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train_unique.csv"
    test_data ="physionet.org/files/vindr-cxr/1.0.0/test"
    test_annotations ="physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv"
    
    all_ims_tr = os.listdir(train_data)
    all_ims_tst = os.listdir(test_data)
    out_dir_tr = 'vindr_jpegs/train/'
    out_dir_test = 'vindr_jpegs/test/'

    classes = {"No finding": 0, 
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
    
    reverse_classes = {v: k for k, v in classes.items()}
    
    class CustomImageDataset(Dataset):
        def __init__(self, data_folder, annotations_file, transform=None):
            self.data_folder = data_folder
            self.annotations_file = annotations_file
            self.transform = transform
            self.image_files = os.listdir(data_folder)
            self.image_files.sort()
            
            # Load annotations from CSV file
            self.annotations = pd.read_csv(annotations_file)
            self.annotations.sort_values(by=['image_id'])
            
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_name = self.annotations.iloc[idx, 0]
            image = Image.open(self.data_folder + img_name + ".jpeg")      
            
            if self.transform:
                image = self.transform(image)
            
            boxes = [];
            areas = [];
            labels = [];
            iscrowd = [];
            target = {
                "boxes": torch.zeros(len(classes), 4),
                "labels": labels,
                "image_id": idx+1,
                "area": torch.zeros(len(classes), dtype=torch.float),
                "iscrowd": iscrowd
            }
            row = idx+1
            for row in range(len(self.annotations)):
                if self.annotations.iloc[row, 0] == img_name:
                    class_name = self.annotations.iloc[row, 1]
                    label = classes.get(class_name)
                    labels.append(label)
                    iscrowd.append(0)
                    if label > 0:
                        x_min = self.annotations.iloc[row, 2]
                        y_min = self.annotations.iloc[row, 3]
                        x_max = self.annotations.iloc[row, 4]
                        y_max = self.annotations.iloc[row, 5]
                    else:  # No finding class - boxes contain entire image
                        x_min = 0
                        y_min = 0
                        x_max = image.shape[2]
                        y_max = image.shape[1]
                    area = (x_max - x_min) * (y_max - y_min)
                    boxes.append([x_min, y_min, x_max, y_max])
                    areas.append(area)   
                else:
                    if len(boxes) > 0:
                        break;
    
            boxes = np.array(boxes)
            target["boxes"] = torch.from_numpy(boxes)
            target["area"] = torch.tensor(areas, dtype=torch.float)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)
            
            return image, target
            
    # Define transform 
    transform = transforms.Compose([
            #transforms.Resize(256),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
    ])
    
    # Create dataset instances for training and testing
    train_dataset = CustomImageDataset(out_dir_tr, train_annotations_unique, transform=transform)
    test_dataset = CustomImageDataset(out_dir_test, test_annotations, transform=transform)
    
    # Create data loaders
    import utils
    from torch.utils.data import DataLoader
    loaders = {
        'train' : DataLoader(train_dataset,
                                              batch_size=6,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=1,
                                              collate_fn=utils.collate_fn),
    
        'test'  : DataLoader(test_dataset,
                                              batch_size=6,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=1,
                                              collate_fn=utils.collate_fn),
    }
    
    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(classes)
    
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes)
    
    #backbone = torchvision.models.swin_t(weights="DEFAULT").features
    class Swin(nn.Module):
      def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.swin_t(weights="DEFAULT").features
        self.out_channels = 768
    
      def forward(self, x):
        return torch.permute(self.backbone(x), (0, 3, 1, 2))
        
    backbone = Swin()
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=num_classes,
        sampling_ratio=4,
    )
    
    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    
    model.to(device)
    
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
    os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")
    
    from engine import train_one_epoch, evaluate
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=0.001,
        #momentum=0.9,
        weight_decay=0.0005
    )
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    num_epochs = 5
    print(f'Training started: {time.strftime("%y%m%d_%H%M%S")}')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loaders["train"], device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, loaders["test"], device=device)
        model_wts = copy.deepcopy(model.state_dict())
        
        save_as = f'swin_fastercnn_model_{time.strftime("%y%m%d_%H%M%S")}.pth'
        torch.save(model_wts, save_as)

if __name__ == '__main__':
    main()
