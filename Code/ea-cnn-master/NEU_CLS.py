import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# 使用 DataLoader 进行批量读取数据
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

ifDebug = True


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

    if ifDebug:
        print("Path Direcorty: ", os.listdir(path2dataset))
        print("Train Direcorty: ", os.listdir(train_dir))
        print("Test Direcorty: ", os.listdir(test_dir))
        print("Validation Direcorty: ", os.listdir(val_dir))

        print("Training Inclusion data:", len(os.listdir(train_dir + '/' + 'Inclusion')))
        print("Testing Inclusion data:", len(os.listdir(test_dir + '/' + 'Inclusion')))
        print("Validation Inclusion data:", len(os.listdir(val_dir + '/' + 'Inclusion')))

    # 定义数据变换
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if cls == "train":
        dl = NEUDataset(data_dir=train_dir, transform=train_transform)
    elif cls == "Validation":
        dl = NEUDataset(data_dir=val_dir, transform=val_transform)
    elif cls == "test":
        dl = NEUDataset(data_dir=test_dir,transform=val_transform)
    else:
        print("!!!!!!   Wrong cls   !!!!!!")
        dl = NEUDataset(train_dir)

    return dl

def get_neucls_dataloader(relative_path="../data", cls="train", bs=32, augment=True,
                          random_seed=42, valid_size=0.2, shuffle=False, show_sample=False,
                          num_workers=1, pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the neucls dataset. A sample
    9x9 grid of the images can be optionally displayed.
    * If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    thisDataset = get_data(relative_path, cls)


    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return DataLoader(thisDataset, batch_size=bs, shuffle=shuffle, pin_memory=pin_memory)


def convert_label_2_numbers(label_string):
        nameDict = {'Crazing': 0, 'Inclusion': 1, 'Patches': 2, 'Pitted': 3, 'Rolled': 4, 'Scratches': 5}
        return int(nameDict[label_string])

class NEUDataset(Dataset):
    """
    CustomDataset 类继承了 PyTorch 的 Dataset 类，它是一个抽象类，用于表示数据集。
        __init__ 函数初始化数据集的参数，包括数据集所在目录、数据变换等。
        __len__ 函数返回数据集的长度，即数据集中图像的数量。
        __getitem__ 函数用于获取指定索引位置的数据。它首先计算出该索引所对应的标签和图像路径，然后打开图像并进行变换，最后返回图像和标签。
            注意，这里的标签需要转换为 PyTorch 中的 Tensor 对象。

    """
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = os.listdir(self.data_dir)

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.data_dir)])

    def __getitem__(self, index):

        # 这里报错了
        # label = self.labels[index // len(os.listdir(self.data_dir))]
        label = self.labels[index]
        label_dir = os.path.join(self.data_dir, label)
        image_path = os.path.join(label_dir, os.listdir(label_dir)[index % len(os.listdir(label_dir))])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 注意这里要进tensor 得是数字形式
        return image, torch.tensor(convert_label_2_numbers(label))




# class CustomDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.classes = sorted(os.listdir(self.data_dir))
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
#         self.samples = []
#         for target_class in self.classes:
#             class_dir = os.path.join(self.data_dir, target_class)
#             if not os.path.isdir(class_dir):
#                 continue
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 if os.path.isfile(img_path):
#                     self.samples.append((img_path, self.class_to_idx[target_class]))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         img_path, label = self.samples[index]
#         img = Image.open(img_path )#.convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, torch.tensor(label)


if __name__ == '__main__':
    # management of pathes
    path2data = './data'
    path2data = os.path.abspath(path2data)
    assert os.path.exists(path2data)

    input_size = 32
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

    projectPath = os.path.dirname(os.path.dirname(__file__))
    datasetName = 'NEU Metal Surface Defects Data'
    path2dataset = os.path.join(path2data, datasetName)

    train_dir = os.path.join(path2dataset, 'train')
    val_dir = os.path.join(path2dataset, 'valid')
    test_dir = os.path.join(path2dataset, 'test')

    if ifDebug:
        print("Path Direcorty: ", os.listdir(path2dataset))
        print("Train Direcorty: ", os.listdir(train_dir))
        print("Test Direcorty: ", os.listdir(test_dir))
        print("Validation Direcorty: ", os.listdir(val_dir))

        print("Training Inclusion data:", len(os.listdir(train_dir + '/' + 'Inclusion')))
        print("Testing Inclusion data:", len(os.listdir(test_dir + '/' + 'Inclusion')))
        print("Validation Inclusion data:", len(os.listdir(val_dir + '/' + 'Inclusion')))

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_datasets = datasets.ImageFolder(train_dir, data_transforms['train'])
    neuclsDataset = image_datasets
    # neuclsDataset = NEUDataset(train_dir, transform)

    # print(neuclsDataset.labels)




    # 调用
    # from torchvision import transforms




    # 使用 DataLoader 进行批量读取数据
    # from torch.utils.data import DataLoader

    batch_size = 32
    dataloader = DataLoader(neuclsDataset, batch_size=batch_size, shuffle=True)
    print('done')

    for _, data in enumerate(dataloader, 0):
        print(data)
