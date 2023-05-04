"Completely Automated CNN Architecture Design Based on Blocks" published by TNNLS.


# Usage 
维护一系列.py文件

## utils.py
_get_available_gpu_plain_info()
[x] 修改默认python的运行pid
writeList 中['/usr/lib/Xorg'] 修改GPU白名单

## evaluate.py
releaseGPUTime 调整等待时间
rGPUTime

## spliteRawNEUCLS.py
分割数据集

### indi0000.py
用于测试单次代码运行

# todo
## 修改到可以使用
- [x] 数据加载器
- [x] 训练文件neucls.py
- [x] 有错误from NEU_CLS import get_neucls_dataloader
- [ ] 推理时候把最后的置信度向量保存下来，用于t-sne降维
- [ ] 利用降维提供一个指标，衡量区分度，多加几个指标

# 可视化
## visdom
```
python3 -m visdom.server
```

# 内容解读
## Completely Automated CNN Architecture Design Based on Block
这篇论文主要用GA算法来进行选择最优的CNN网络结构，AE-CNN完全自动化的网络结构设计。
这里设计的CNN中有ResNet Block, DenseNet Block,Pooling layer，不包括全连接层。
GA算法主要需要考虑的有以下几个方面：

对个体进行编码（individual encode）
适应度（fitness evaluation）
配种选择策略（mating selection）
交叉（crossover）和基因突变（mutation）
环境选择（environment election）
停止准则（stoping crieterion）


## 文章中用到的策略
### 一、种群初始化
population.py
主要是基于采用的个体编码策略，本篇文章中采用变长编码，因为设计的CNN的depth可变。这里的编码策略基于三种不同类型的units以及它们在网络中对应的position。RBU,DBU,PU分别是ResNet Unit,DenseNet Unit,Pooling Unit,其中一个RBU，DBU可以包括多个RBs,DBs,而PU中仅包含有一个pooling layer。其中的RB，DB中的convolution layers的数目是用户确定的。
RBU的parameter：position，input，output，type，amount。其中的amount指的是the number of convolutional layers of the unit。
DBU同上，但多出一个k
```
class Individual(object):
def initialize(self):
```
acc： -1 表示清零了
PU：position，type of the unit，the pooling type
种群的初始化两重循环：遍历P0中的每个个体；为每个个体生成编码信息，并用一个数组进行存放。
初始化的种群P0包含N个长度不一的数组，即对应N个个体，或者说是N个候选的CNN结构。
### 二、适应度
由于该CNN用于图像分类，所以classification accuracy作为individual fitness。要想知道每个个体的fitness，就需要将其编码信息decode成对应的CNN architecture，然后权重初始化,然后拿training data对该网络进行训练，网络参数的优化采用SGDM，优化完之后拿validation data来验证该网络模型的分类准确性，并将此accuracy作为对应“个体”的fitness。
### 三、配种选择
是指从父代群体Pt中选出parent，用以产生offspring。本算法中采用的是binary tournament select，先随机选出两个个体，然后比较其适应度，适应度较大的做parent，同理，总共选出两个个体作为双亲，产生两个offsprings。
### 四、交叉、基因突变
交叉率，突变率是事先确定好的数值。
交叉算子是GA的核心。
交叉—局部搜索----exploit
基因突变----全局搜索----explore
由于这里采用的变长编码技术，所以one-point crossover operator，来确定断裂重组的位置。
突变又分为三种：增，删，改
在执行交叉变换和基因突变的时候要注意code的正确性，即需要保证position的连续，该层的输入等于上一层的输出。
### 五、环境选择
从Pt与Qt中保留N个个体作为下一代的父代Pt+1。Pt是父代，Qt是生成的子代，Pt+Qt当做当前代，共有2N个个体。保留策略：2N个个体中适应度最高的个体无条件地保留加入到Pt+1中，剩下的N-1个个体用binary tournament selection。。采用binary tournament selection主要是考虑到环境选择时既要保证群体的convergence，又要diversity。如果只保留适应度最高的个体，可能多样性就保证不了了，从而容易陷入局部最优和过早收敛。
### 六、停止准则
这里采用设置一定的迭代次数maximal generation number为20，即进化20代停止。

## GAs中的选择策略：

Proportionate Roulette Wheel Selection 轮盘赌选择
tournament selection 锦标赛选择
linear ranking selection
exponential ranking selection
其中的轮盘赌选择法有一个缺点，如果有的个体的适应度为0，那么他们被选中的可能性为0.
后两种选择法需要将种群中的个体安装适应值进行排序。

## 网络结构优化方法：
SGD—>SGD with momentum–>-->nesterov momentum—>AdaGrad–>AdaDelta–>RMSProp–>Adam
SGD容易陷入局部最优，容易被鞍点困住，因为鞍点附近的梯度均接近0，没法更新参数。
SGDM:SDG+momentum，历史梯度与当前梯度结合,将SGD与梯度的一阶矩估计结合
nesterov momentum:不是对当前批求梯度，而是对θ-γ * v求导, 而不是原始的θ
AdaGrad,adadelta，RMSProp引入梯度的二阶矩估计，可以使学习率在训练的过程中自动地改变。
Adam利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。

# 结

配种选择和环境选择都是基于所有个体适应度已知的前提下进行的。
在配种选择和环境选择中都采用了binary tournament selection。
交叉—局部搜索，基因突变----全局搜索；环境选择时convergence和diversity都要兼顾。选择算法时要做好平衡。
由于每一轮的mate产生新的子代Q时，都需要计算Q中所有个体的适应度，而每次适应度的计算都相当于对某一特定结构的网络的训练，即假设总共进化20代，种群量为20，那么需要对400个网络进行训练，这样耗时较长，所以加速对个体适应度的评估是有必要的。
