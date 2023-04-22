"""
2023-04-07  21:31:47
"""
from __future__ import print_function
import torch,torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_loader
import os
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool

# 有错误
# from NEU_CLS import get_neucls_dataloader
# 新的
from NEU_CLS_dataloader import get_neucls_dataloader

# 临时添加的
from utils import Utils,GPUTools
import importlib
from multiprocessing import Process
import time, os, sys



class SurfaceDefectResNet(torch.nn.Module):

    def __init__(self):
        super(SurfaceDefectResNet, self).__init__()
        self.cnn_layers = torchvision.models.resnet34(pretrained=True)
        num_ftrs = self.cnn_layers.fc.in_features
        self.cnn_layers.fc = torch.nn.Linear(num_ftrs, 6)

    def forward(self, x):
        # stack convolution layers
        out = self.cnn_layers(x)
        return out


"""torchvision.models.可用的网络
https://pytorch.org/vision/stable/models.html
AlexNet
ConvNeXt
DenseNet
EfficientNet
EfficientNetV2
GoogLeNet
Inception V3
MaxVit
MNASNet
MobileNet V2
MobileNet V3
RegNet
ResNet
ResNeXt
ShuffleNet V2
SqueezeNet
SwinTransformer
VGG
VisionTransformer
Wide ResNet
"""


class SurfaceDefectDenseNet(torch.nn.Module):

    def __init__(self):
        super(SurfaceDefectDenseNet, self).__init__()
        self.cnn_layers = torchvision.models.DenseNet(num_classes=6)
        # num_ftrs = self.cnn_layers.fc.in_features
        # self.cnn_layers.fc = torch.nn.Linear(num_ftrs, 6)

    def forward(self, x):
        # stack convolution layers
        out = self.cnn_layers(x)
        return out


class SurfaceDefectSwin(torch.nn.Module):

    def __init__(self):
        super(SurfaceDefectSwin, self).__init__()
        '''
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        '''
        # self.cnn_layers = torchvision.models.swin_transformer.SwinTransformer([3,3], 3, [1,1], [1,1], [1,1])
        self.cnn_layers = torchvision.models.swin_transformer.swin_t()
        # num_ftrs = self.cnn_layers.fc.in_features
        # self.cnn_layers.fc = torch.nn.Linear(num_ftrs, 6)

    def forward(self, x):
        # stack convolution layers
        out = self.cnn_layers(x)
        return out


class TrainModel(object):
    def __init__(self):
        # 目录所引到"cifar-10-batches-py"
        # trainloader, validate_loader = data_loader.get_train_valid_loader('../data/',
        #         batch_size=128, augment=True, valid_size=0.1, shuffle=False, random_seed=2312390,
        #         show_sample=False, num_workers=2, pin_memory=True)
        # trainloader = get_neucls_dataloader(relative_path='../data', cls="train")
        # validate_loader = get_neucls_dataloader(relative_path='../data', cls="Validation")
        loader = get_neucls_dataloader(data_dir = '../data/NEU-CLS/', num_epochs = 20, batch_size = 8, input_size = 200, num_workers=1, shuffle=True)
        trainloader = loader['train']
        validate_loader = loader['val']
        # testloader = data_loader.get_test_loader('/home/yanan/train_data', batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        # /tmp/pycharm_project_663/genetic
        # net = EvoCNNModel()
        # net = SurfaceDefectResNet()
        # net = SurfaceDefectDenseNet()
        net = SurfaceDefectResNet()
        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.trainloader = trainloader
        self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]
        #self.testloader = testloader
        #self.log_record(net, first_time=True)
        #self.log_record('+'*50, first_time=False)

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch):
        self.net.train()
        print("epoch:", epoch)
        lr = 0.001
        if epoch ==0: lr = 0.01
        # if epoch > 0: lr = 0.1;
        if epoch < 25: lr = 0.001
        # if epoch > 40: lr = 0.001
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-4)
        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.trainloader, 0):
            # 这里会报一个错
            print(".", end="")
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # print("进了cuda")
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        # 4.11 卡死在这里
        print('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))
        # self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct/total > self.best_acc:
            self.best_acc = correct/total
            print('*'*100, self.best_acc)
        # self.log_record('Validate-Loss:%.3f, Acc:%.3f'%(test_loss/total, correct/total))


    def process(self):
        # os.path.append("../")
        total_epoch = StatusUpdateTool.get_epoch_size()
        print('Total epoch:', total_epoch)
        for p in range(total_epoch):
            self.train(p)
            self.test(total_epoch)
        return self.best_acc


class RunModel(object):
    def do_workk(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        m = TrainModel()
        try:
            # m = TrainModel()
            # m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            best_acc = m.process()
            print('done  best_acc = m.process() ')
            #import random
            #best_acc = random.random()
        except BaseException as e:
            # 找不到global.ini 的 [network]
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            # m.log_record('Exception occur:%s'%(str(e)))
        finally:
            # m.log_record('Finished-Acc:%.3f'%best_acc)
            print('finally')
            # f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            # f.write('%s=%.5f\n'%(file_id, best_acc))
            # f.flush()
            # f.close()

if __name__ == '__main__':
    #gpu_id = GPUTools.detect_availabel_gpu_id()
    gpu_id = str(1) # 报错一次
    file_name = 'indi0000'
    r = RunModel()
    r.do_workk(gpu_id=gpu_id, file_id=file_name)