import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import set_seed
from trainer import *

set_seed(42)

"""Data 2"""
data_dir = "./data/NEU-CLS/"
# pretrained_path = "models/best_resnet18_ImageNet_NEU-64.pth"
# pretrained_path = "models/best_resnet18_ImageNet.pth"
# pretrained_path = "models/best_resnet18_NEU-64.pth"
num_classes = 6  # for NEU-CLS-200
model_name = "resnet_NEU-64"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

input_size = 224
batch_size = 8
lr = 0.0001
num_epochs = 25

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
# model_ft = model_ft.to(device)