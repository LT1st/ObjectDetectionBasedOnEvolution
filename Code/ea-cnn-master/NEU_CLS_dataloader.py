import os
import torch

import numpy as np
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from utils import set_seed
from torch.utils.data import Dataset
import random
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
ifDebug =True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


set_seed(42)


def get_data(relative_path="./data", cls = "train"):
    # management of relative_path
    path2data = relative_path
    path2data = os.path.abspath(path2data)
    if ifDebug:
        print(path2data)
    assert os.path.exists(path2data)
    # projectPath = os.path.dirname(os.path.dirname(__file__))
    datasetName = 'NEU Metal Surface Defects Data'
    path2dataset = os.path.join(path2data, datasetName)

    train_dir = os.path.join(path2dataset, 'train')
    val_dir = os.path.join(path2dataset, 'valid')
    test_dir = os.path.join(path2dataset, 'test')

def get_neucls_dataloader(data_dir = "./data/NEU-CLS/", num_epochs = 25, batch_size = 16, input_size = 32):

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
    # train_image_datasets =  datasets.ImageFolder(os.path.join(data_dir, 'train'))
    # val_image_datasets =  datasets.ImageFolder(os.path.join(data_dir, 'val'))
    # Create training and validation dataloaders
    # 问题应该是这里返回的不是 batchsize*channel*w*h
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
        # x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        # ['train', 'val']}

    return dataloaders_dict
# Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
# model_ft = model_ft.to(device)

if __name__ == '__main__':
    dl = get_neucls_dataloader()
    a = dl['train']
    for batchNow, dataNow in enumerate(a):
        print(0)