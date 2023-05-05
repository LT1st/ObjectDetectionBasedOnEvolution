"""
2023-05-02  17:24:32
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_loader
# from data.dataloader.NEU_CLS import get_neucls_dataloader
from NEU_CLS_dataloader import get_neucls_dataloader
import os
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool

ifDebug = True          # 是否打印debug信息
ifLog = True            # 是否把信息记录到log文件中  当单次运行或debug时需要注意
ifPrintMemory = True    #

class ResNetBottleneck(nn.Module):
    # expansion 是一个用于扩展通道数的参数，用于控制每个 ResNet 模块中的卷积层输出通道数相对于输入通道数的倍数。
    # 当 expansion = 1 时，表示卷积层输出的通道数与输入通道数相同，不发生通道数的变化
    # ResNet 的 Bottleneck 模块中，expansion 的值通常设置为 4，即输出通道数是输入通道数的 4 倍。
    # 这样做的目的是为了在网络加深时保持较小的模型参数量和计算复杂度，同时提升网络的表达能力，从而获得更好的性能。
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        # 定义第一个卷积层，使用 1x1 的卷积核，用于降低输入平面的维度
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # 定义 conv1 后的批归一化层，用于规范化卷积层的输出
        self.bn1 = nn.BatchNorm2d(planes)
        # 用于提取特征并调整输入尺寸
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 定义第三个卷积层，使用 1x1 的卷积核，用于恢复输出平面的维度
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # 如果 stride 不等于 1，或者输入的平面数不等于扩展倍数乘以输出平面数，
        # 则将 shortcut 定义为一个包含一个 1x1 的卷积层和批归一化层的序列模块，
        # 用于调整输入尺寸或维度，以便与卷积层的输出相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetUnit(nn.Module):
    # 标准的 ResNet 网络结构定义代码通常包含多个 ResNet 模块，并且通常在每个模块中会使用不同的通道数和步幅。
    # 在初始化时调用 _make_layer 方法，创建包含多个 ResNetBottleneck 层的层序列（nn.Sequential），并将其保存在 self.layer 中。
    # _make_layer 方法根据输入的 block 类型、输出通道数 planes、ResNetBottleneck 层的数量 num_blocks 和步长 stride，
    # 生成一个包含多个 ResNetBottleneck 层的层序列（nn.Sequential）。

    def __init__(self, amount, in_channel, out_channel):
        super(ResNetUnit, self).__init__()
        self.in_planes = in_channel
        self.layer = self._make_layer(ResNetBottleneck, out_channel, amount, stride=1)      # 创建ResNetBottleneck层

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)     # 定义每个ResNetBottleneck层的步长，第一个步长为stride，其余为1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 添加ResNetBottleneck层到layers列表
            self.in_planes = planes * block.expansion  # 更新输入通道数，乘以ResNetBottleneck的扩展因子
        # *layers 将 layers 列表中的多个元素展开，并作为参数传递给 nn.Sequential() 函数的构造函数
        return nn.Sequential(*layers)  # 返回包含所有ResNetBottleneck层的Sequential层

    def forward(self, x):
        out = self.layer(x)  # 前向传播，将输入x传递给ResNetBottleneck层
        return out  # 返回输出

class DenseNetBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        # nChannels       nChannels 是 DenseNetBottleneck 模块中输入特征的通道数。在该模块中，nChannels 是输入特征 x 的通道数，
        # 即 x 的 shape[1]，表示输入特征图的通道数。在模块内部，nChannels 经过卷积操作后会发生变化，被缩小到 interChannels，
        # 然后又通过第二个卷积层扩展回 growthRate。
        # growthRate      输出特征图通道数的增加量，通常设置为一个较小的值，如 12 或 24，用于控制网络的模型大小和计算复杂度。
        super(DenseNetBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseNetUnit(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel):
        #     max_input_channel max_input_channel 是指在 DenseNetUnit 中，输入通道数的最大限制。如果输入通道数 in_channel 大于
        # max_input_channel，则会对输入进行 1x1 卷积操作，将输入通道数减少到 max_input_channel。这样可以限制输入通道数的最大值，
        # 从而控制模型的复杂度和计算资源的消耗。在代码中，max_input_channel 用于判断是否需要进行 1x1 卷积的条件判断。

        super(DenseNetUnit, self).__init__()
        self.out_channel = out_channel
        if in_channel > max_input_channel:
            self.need_conv = True
            self.bn = nn.BatchNorm2d(in_channel)
            self.conv = nn.Conv2d(in_channel, max_input_channel, kernel_size=1, bias=False)
            in_channel = max_input_channel

        self.layer = self._make_dense(in_channel, k, amount)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        #
        # Parameters
        # ----------
        # nChannels       nChannels 表示输入特征图的通道数
        #                 在每个稠密块中，nChannels 的值会随着每个层中的特征图通道数的增加而累积，从而得到最终的输出通道数。
        #                 这种累积的方式使得 DenseNet 中的每个层都可以直接访问之前层的特征
        # growthRate      growthRate 表示每个稠密块（Dense Block）中的输出通道数（即每个稠密连接的通道数）。
        #                 每个稠密块中的每个层都将产生 growthRate 个特征图作为输出，并作为下一层的输入。
        #                 这种设计可以帮助网络更加充分地利用之前层的特征，从而增强特征传递和梯度流动，提高网络性能。
        #                 growthRate 是 DenseNet 中的一个超参数，可以根据具体任务和需求进行调整。
        # nDenseBlocks    每个 DenseNet 单元（DenseNet Unit）中包含的稠密块（Dense Block）的数量。
        #                 超参数，控制了网络的深度和复杂度。较大的 nDenseBlocks 值可以增加网络的深度，从而提供更强大的特征提取能力，
        #                 但也会增加网络的计算复杂度和参数量。
        # -------
        #
        layers = []
        for _ in range(int(nDenseBlocks)):
            layers.append(DenseNetBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        # 如果需要进行1x1卷积，则先应用BN和ReLU，再进行卷积
        if hasattr(self, 'need_conv'):      # 判断对象 self 是否具有名为 'need_conv' 的属性
            out = self.conv(F.relu(self.bn(out)))
        out = self.layer(out)
        assert(out.size()[1] == self.out_channel)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self, in_size=200, num_class=6, input_channel=3):
        super(EvoCNNModel, self).__init__()

        #resnet and densenet unit
        self.op0 = DenseNetUnit(k=40, amount=4, in_channel=3, out_channel=163, max_input_channel=32)
        self.op2 = DenseNetUnit(k=20, amount=7, in_channel=163, out_channel=204, max_input_channel=64)
        self.op3 = DenseNetUnit(k=20, amount=7, in_channel=204, out_channel=204, max_input_channel=64)
        self.op4 = ResNetUnit(amount=8, in_channel=204, out_channel=64)
        self.op8 = ResNetUnit(amount=3, in_channel=64, out_channel=256)
        self.op9 = DenseNetUnit(k=12, amount=6, in_channel=256, out_channel=200, max_input_channel=128)

        #linear unit
        self.linear = nn.Linear(28800, 6)


    def forward(self, x):
        out= self.op0(x)
        out = F.avg_pool2d(out, 2)
        out= self.op2(out)
        out= self.op3(out)
        out = self.op4(out)
        out = F.avg_pool2d(out, 2)
        out = F.avg_pool2d(out, 2)
        out= F.max_pool2d(out, 2)
        out = self.op8(out)
        out= self.op9(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self):
        # 用于训练其他数据集
        #trainloader, validate_loader = data_loader.get_train_valid_loader('./data', batch_size=128, augment=True, valid_size=0.1, shuffle=False, random_seed=2312390, show_sample=False, num_workers=4, pin_memory=True)
        #testloader = data_loader.get_test_loader('/home/yanan/train_data', batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        # 从globel.ini获取参数
        input_size = StatusUpdateTool.get_input_size()
        num_class = StatusUpdateTool.get_num_class()
        input_channel = StatusUpdateTool.get_input_channel()
        this_epoch = StatusUpdateTool.get_epoch_size()
        this_batch_size = StatusUpdateTool.get_batch_size()

        # loader = get_neucls_dataloader(data_dir = './data/NEU-CLS/', num_epochs = 16, batch_size = 8, input_size = 200, num_workers=1, shuffle=True)
        loader = get_neucls_dataloader(data_dir = './data/NEU-CLS/', num_epochs = this_epoch,
                                       batch_size = this_batch_size, input_size = input_size)
        trainloader = loader['train']
        validate_loader = loader['val']
        net = EvoCNNModel(in_size=input_size, num_class=num_class, input_channel=input_channel)
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
        if ifLog:
            dt = datetime.now()
            dt.strftime( '%Y-%m-%d %H:%M:%S' )
            if first_time:
                file_mode = 'w'
            else:
                file_mode = 'a+'
            f = open('./log/%s.txt'%(self.file_id), file_mode)
            f.write('[%s]-%s\n'%(dt, _str))
            f.flush()
            f.close()
        else:
            print(_str)

    def train(self, epoch):
        if ifDebug:
            print(epoch, end = "--")
        self.net.train()
        lr = 0.0001
        # if epoch <= 3: lr = 0.01
        # if epoch > 10: lr = 0.1;
        # if epoch > 25: lr = 0.01
        # if epoch > 40: lr = 0.001
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-4)
        running_loss = 0.0
        total = 0
        correct = 0
        for mini_batch_cnt, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            if mini_batch_cnt == 1 and ifPrintMemory:
                print("train outputs:{}".format(torch.cuda.memory_allocated(0)/1024/1024))
            loss = self.criterion(outputs, labels)
            loss.backward()
            if mini_batch_cnt == 1 and ifPrintMemory:
                print("train backward:{}".format(torch.cuda.memory_allocated(0)/1024/1024))
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if ifDebug:
            print("going to record", end = "++")
            print('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))
        torch.cuda.empty_cache()
        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for mini_batch_cnt, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            if mini_batch_cnt==2 and ifPrintMemory:
                print("validate outputs:{}".format(torch.cuda.memory_allocated(0)/1024/1024), mini_batch_cnt)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if mini_batch_cnt==2 and ifPrintMemory:
                print("validate total:{}".format(torch.cuda.memory_allocated(0)/1024/1024), mini_batch_cnt)
            correct += (predicted == labels.data).sum()
        if correct/total > self.best_acc:
            self.best_acc = correct/total
            print('*'*100, self.best_acc)
        self.log_record('Validate-Loss:%.3f, Acc:%.3f'%(test_loss/total, correct/total))


    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        for p in range(total_epoch):
            self.train(p)
            self.test(total_epoch)
        return self.best_acc


class RunModel(object):
    def do_workk(self, gpu_id, file_id):
        if ifDebug:
            print("do_workk", end = "**")
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        m = TrainModel()
        try:
            # m = TrainModel()
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            best_acc = m.process()
            #import random
            #best_acc = random.random()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.3f'%best_acc)

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s=%.5f\n'%(file_id, best_acc))
            f.flush()
            f.close()