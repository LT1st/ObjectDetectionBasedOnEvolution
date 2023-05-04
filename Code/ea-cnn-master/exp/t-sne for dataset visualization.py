from NEU_CLS_dataloader import get_neucls_dataloader
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# path2data = '../data'
# path2data = os.path.abspath(path2data)
# assert os.path.exists(path2data)
#
# projectPath = os.path.dirname(os.path.dirname(__file__))
# datasetName = 'NEU Metal Surface Defects Data'
# path2dataset = os.path.join(path2data, datasetName)
#
# train_dir = os.path.join(path2dataset, 'train')
# val_dir = os.path.join(path2dataset, 'valid')
# test_dir = os.path.join(path2dataset, 'test')
#
# image_datasets = datasets.ImageFolder(train_dir, data_transforms['train'])
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 加载数据集
train_loader = get_neucls_dataloader(data_dir = '../data/NEU-CLS/', num_epochs = 16,
                               batch_size = 32, input_size = 200)
# batch_size = 100
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
all = train_loader['train']#+train_loader['val']
# 从DataLoader中获取数据
# 从DataLoader中获取所有数据
data = []
labels = []
for batch_data, batch_labels in all:
    data.append(batch_data.view(batch_data.shape[0], -1).numpy())
    labels.append(batch_labels.numpy())
data = np.concatenate(data)
labels = np.concatenate(labels)

# data, labels = next(iter(all))
# data = data.view(data.shape[0], -1).numpy()
# labels = labels.numpy()

# 对数据进行归一化
scaler = StandardScaler()
data = scaler.fit_transform(data)

# 对数据进行t-SNE降维
# tsne = TSNE(n_components=2, perplexity=6, n_iter=20000)
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data)

# 绘制可视化图形
# sns.set_palette("bright")
sns.set_palette("Set1")
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    legend="full",
    alpha=0.4
)
plt.title("t-SNE Visualization of Dataset")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()


