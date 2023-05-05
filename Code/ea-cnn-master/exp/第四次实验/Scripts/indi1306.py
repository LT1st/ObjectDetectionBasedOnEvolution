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

ifDebug = True          # �Ƿ��ӡdebug��Ϣ
ifLog = True            # �Ƿ����Ϣ��¼��log�ļ���  ���������л�debugʱ��Ҫע��
ifPrintMemory = True    #

class ResNetBottleneck(nn.Module):
    # expansion ��һ��������չͨ�����Ĳ��������ڿ���ÿ�� ResNet ģ���еľ�������ͨ�������������ͨ�����ı�����
    # �� expansion = 1 ʱ����ʾ����������ͨ����������ͨ������ͬ��������ͨ�����ı仯
    # ResNet �� Bottleneck ģ���У�expansion ��ֵͨ������Ϊ 4�������ͨ����������ͨ������ 4 ����
    # ��������Ŀ����Ϊ�����������ʱ���ֽ�С��ģ�Ͳ������ͼ��㸴�Ӷȣ�ͬʱ��������ı���������Ӷ���ø��õ����ܡ�
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        # �����һ������㣬ʹ�� 1x1 �ľ���ˣ����ڽ�������ƽ���ά��
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # ���� conv1 �������һ���㣬���ڹ淶�����������
        self.bn1 = nn.BatchNorm2d(planes)
        # ������ȡ��������������ߴ�
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # �������������㣬ʹ�� 1x1 �ľ���ˣ����ڻָ����ƽ���ά��
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # ��� stride ������ 1�����������ƽ������������չ�����������ƽ������
        # �� shortcut ����Ϊһ������һ�� 1x1 �ľ���������һ���������ģ�飬
        # ���ڵ�������ߴ��ά�ȣ��Ա��������������
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
    # ��׼�� ResNet ����ṹ�������ͨ��������� ResNet ģ�飬����ͨ����ÿ��ģ���л�ʹ�ò�ͬ��ͨ�����Ͳ�����
    # �ڳ�ʼ��ʱ���� _make_layer ����������������� ResNetBottleneck ��Ĳ����У�nn.Sequential���������䱣���� self.layer �С�
    # _make_layer ������������� block ���͡����ͨ���� planes��ResNetBottleneck ������� num_blocks �Ͳ��� stride��
    # ����һ��������� ResNetBottleneck ��Ĳ����У�nn.Sequential����

    def __init__(self, amount, in_channel, out_channel):
        super(ResNetUnit, self).__init__()
        self.in_planes = in_channel
        self.layer = self._make_layer(ResNetBottleneck, out_channel, amount, stride=1)      # ����ResNetBottleneck��

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)     # ����ÿ��ResNetBottleneck��Ĳ�������һ������Ϊstride������Ϊ1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # ���ResNetBottleneck�㵽layers�б�
            self.in_planes = planes * block.expansion  # ��������ͨ����������ResNetBottleneck����չ����
        # *layers �� layers �б��еĶ��Ԫ��չ��������Ϊ�������ݸ� nn.Sequential() �����Ĺ��캯��
        return nn.Sequential(*layers)  # ���ذ�������ResNetBottleneck���Sequential��

    def forward(self, x):
        out = self.layer(x)  # ǰ�򴫲���������x���ݸ�ResNetBottleneck��
        return out  # �������

class DenseNetBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        # nChannels       nChannels �� DenseNetBottleneck ģ��������������ͨ�������ڸ�ģ���У�nChannels ���������� x ��ͨ������
        # �� x �� shape[1]����ʾ��������ͼ��ͨ��������ģ���ڲ���nChannels �������������ᷢ���仯������С�� interChannels��
        # Ȼ����ͨ���ڶ����������չ�� growthRate��
        # growthRate      �������ͼͨ��������������ͨ������Ϊһ����С��ֵ���� 12 �� 24�����ڿ��������ģ�ʹ�С�ͼ��㸴�Ӷȡ�
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
        #     max_input_channel max_input_channel ��ָ�� DenseNetUnit �У�����ͨ������������ơ��������ͨ���� in_channel ����
        # max_input_channel������������� 1x1 ���������������ͨ�������ٵ� max_input_channel������������������ͨ���������ֵ��
        # �Ӷ�����ģ�͵ĸ��ӶȺͼ�����Դ�����ġ��ڴ����У�max_input_channel �����ж��Ƿ���Ҫ���� 1x1 ����������жϡ�

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
        # nChannels       nChannels ��ʾ��������ͼ��ͨ����
        #                 ��ÿ�����ܿ��У�nChannels ��ֵ������ÿ�����е�����ͼͨ���������Ӷ��ۻ����Ӷ��õ����յ����ͨ������
        #                 �����ۻ��ķ�ʽʹ�� DenseNet �е�ÿ���㶼����ֱ�ӷ���֮ǰ�������
        # growthRate      growthRate ��ʾÿ�����ܿ飨Dense Block���е����ͨ��������ÿ���������ӵ�ͨ��������
        #                 ÿ�����ܿ��е�ÿ���㶼������ growthRate ������ͼ��Ϊ���������Ϊ��һ������롣
        #                 ������ƿ��԰���������ӳ�ֵ�����֮ǰ����������Ӷ���ǿ�������ݺ��ݶ�����������������ܡ�
        #                 growthRate �� DenseNet �е�һ�������������Ը��ݾ��������������е�����
        # nDenseBlocks    ÿ�� DenseNet ��Ԫ��DenseNet Unit���а����ĳ��ܿ飨Dense Block����������
        #                 ���������������������Ⱥ͸��Ӷȡ��ϴ�� nDenseBlocks ֵ���������������ȣ��Ӷ��ṩ��ǿ���������ȡ������
        #                 ��Ҳ����������ļ��㸴�ӶȺͲ�������
        # -------
        #
        layers = []
        for _ in range(int(nDenseBlocks)):
            layers.append(DenseNetBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        # �����Ҫ����1x1���������Ӧ��BN��ReLU���ٽ��о��
        if hasattr(self, 'need_conv'):      # �ж϶��� self �Ƿ������Ϊ 'need_conv' ������
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
        # ����ѵ���������ݼ�
        #trainloader, validate_loader = data_loader.get_train_valid_loader('./data', batch_size=128, augment=True, valid_size=0.1, shuffle=False, random_seed=2312390, show_sample=False, num_workers=4, pin_memory=True)
        #testloader = data_loader.get_test_loader('/home/yanan/train_data', batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        # ��globel.ini��ȡ����
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