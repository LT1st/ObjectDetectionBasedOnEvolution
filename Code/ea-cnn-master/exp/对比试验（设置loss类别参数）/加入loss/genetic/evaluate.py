from utils import Utils, GPUTools, GPUStatus
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep

ifDebug = True

import re
import numpy as np
from scipy.special import expit
import os


def fill_zero_values_with_average(data_dict):
    non_zero_values = [value for value in data_dict.values() if value != 0]  # 获取非零值
    average = sum(non_zero_values) / len(non_zero_values)  # 计算非零值的平均值

    filled_dict = {key: value if value != 0 else average for key, value in data_dict.items()}  # 填充零值为平均值

    return filled_dict


def get_loss_acc_from_file(path_2_file):
    # 读取 log_xx.txt 文件
    with open(path_2_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_process_2 = []  # 用于保存提取的 Train-Epoch、Loss、Acc 数组

    # 遍历每一行
    for line in lines:
        # 使用正则表达式匹配 Train-Epoch、Loss、Acc 数字
        match = re.search(r'Train-Epoch:\s*(\d+).*Loss:\s*([\d.]+).*Acc:\s*([\d.]+)', line)
        if match:
            # 提取 Train-Epoch、Loss、Acc 数字并转换为 float 类型
            train_epoch = int(match.group(1))
            # 将 "NaN" 转换为 -1
            loss = float(match.group(2)) if match.group(2) != "NaN" else -1
            acc = float(match.group(3)) if match.group(3) != "NaN" else -1
            # 将 Train-Epoch、Loss、Acc 数字放入 tmp 数组中
            tmp = [train_epoch, loss, acc]
            # 将 tmp 数组放入 train_process 数组中
            train_process_2.append(tmp)

    return train_process_2


def sigmoid_normalize_column(arr):
    column = np.array(arr)[:, 1]  # Extract the second column data
    normalized_column = expit(column)  # Use expit function for sigmoid normalization
    arr = np.array(arr)  # 将输入的列表转换为NumPy数组
    arr[:, 1] = normalized_column  # Replace the original array's second column with the normalized values
    return arr.tolist()


def custom_normalize_column(arr, threshold=1):
    column = np.array(arr)[:, 1]  # 提取第二列数据
    column[column > threshold] = 1  # 大于阈值的值设为1
    # column = (column - np.min(column)) / (np.max(column) - np.min(column))  # 线性归一化计算
    arr = np.array(arr)  # 将输入的列表转换为NumPy数组
    arr[:, 1] = column  # 将归一化后的值替换原数组的第二列
    return arr.tolist()

# array = np.array([[1, 0.842, 0.452], [2, 0.939, 0.636]])
# normalized_array = custom_normalize_column(array)
# print(normalized_array)


def pred_acc_(all_acc):
    # 使用模型预测acc
    return 0.1


def read_finished_acc_values(data_dict):
    log_folder = "./log"
    query_dict = {}

    for key, _ in data_dict.items():
        filename = os.path.join(log_folder, key + ".txt")
        print(filename)
        if os.path.isfile(filename):
            try:
                this_acc = get_loss_acc_from_file(filename)
                this_acc= sigmoid_normalize_column(this_acc)
                # print(this_acc)
                pred_acc = pred_acc_(this_acc)

                all_acc = this_acc[-1][2]*0.9 + this_acc[-1][1]*0.1 + pred_acc
                print(all_acc)
                query_dict[key] = all_acc
            except:
                query_dict[key] = 0
                # pass

        else:
            query_dict[key] = 0
            # pass

    return query_dict


class FitnessEvaluate(object):

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        """
        生成对应子代模型的代码
        Returns
        -------

        """
        self.log.info('Begin to generate python files in evaluate.py line:16')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)   # bug
        self.log.info('Finish the generation of python files')

    def evaluate(self):
        """
        load fitness from cache file
        """
        mutiGPU = True     # 多卡卡死
        # GPU_status
        gpu_status = GPUStatus()
        print(gpu_status.gpu_num)
        self.log.info('Try to query fitness from cache from ./populations/cache.txt')
        # 已有网络结构
        _map = Utils.load_cache_data()
        _count = 0 # 有多少数据已经有了
        for indi in self.individuals:
            # 个体编号  个体网络结构
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _acc = _map[_key]
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f'%(indi.id, _key, float(_acc), indi.acc))
                indi.acc = float(_acc)
        # 有多少已有数据
        self.log.info('Total hit %d individuals for fitness'%(_count))

        # 睡眠时间定义
        releaseGPUTime = int(10)  # 60
        rGPUTime = int(5)  # 30

        has_evaluated_offspring = False
        for indi in self.individuals:
            if indi.acc < 0:  # 如果acc小于0，说明没训练过
                if not mutiGPU:
                    has_evaluated_offspring = True
                    """ 
                    # 测试单卡单次
                    has_evaluated_offspring = True
                    file_name = indi.id
                    self.log.info('Begin to train %s'%(file_name))
                    _module = importlib.import_module('.', 'Scripts.%s'%(file_name))
                    _class = getattr(_module, 'RunModel')
                    cls_obj = _class()
                    cls_obj.do_work('1', file_name)
                    """
                    # GPU状态维护
                    # time.sleep(releaseGPUTime)

                    gpu_id = GPUTools.detect_if_gpu_0_availabel()
                    while gpu_id is None:
                        time.sleep(rGPUTime)
                        gpu_id = GPUTools.detect_if_gpu_0_availabel()
                    if gpu_id is not None:
                        file_name = indi.id
                        self.log.info('Begin to train %s' % (file_name))
                        module_name = 'Scripts.%s' % (file_name)

                        # 从文件加载模型
                        """
                        sys.modules将自动记录该模块。当第二次再导入该模块时，python会直接到字典中查找，从而加快了程序运行的速度。
                        字典sys.modules具有字典所拥有的一切方法，可以通过这些方法了解当前的环境加载了哪些模块
                        """
                        if module_name in sys.modules.keys():
                            self.log.info('Module:%s has been loaded, delete it' % (module_name))
                            del sys.modules[module_name]
                            _module = importlib.import_module('.', module_name)
                        else:
                            _module = importlib.import_module('.', module_name)
                        # 访问对象的属性值，并提供在密钥不可用的情况下执行默认值的选项
                        _class = getattr(_module, 'RunModel')
                        cls_obj = _class()
                        # BUG 这里gpuid问题
                        p = Process(target=cls_obj.do_workk, args=(str(gpu_id), file_name,))
                        # p = Process(target=cls_obj.do_workk, args=('%d'%(gpu_id), file_name,))
                        p.start()
                        # has_evaluated_offspring = True
                        # 不能直接读取，得等他训练完了，不然读取到的就是以前的
                        # indi.acc = Utils.get_acc_by_id(file_name)

                # 多卡训练
                else:
                    if ifDebug:
                        print("muti_gpu")
                    has_evaluated_offspring = True

                    time.sleep(releaseGPUTime)
                    # GPU状态维护
                    gpu_id =gpu_status.get_an_available_gpu()
                    # gpu_id = GPUTools.detect_availabel_gpu_id()
                    while gpu_id is None:
                        time.sleep(rGPUTime)
                        # gpu_id = GPUTools.detect_availabel_gpu_id()
                        gpu_id = gpu_status.get_an_available_gpu()
                    if gpu_id is not None:
                        print("is using gpu id",gpu_id)
                        file_name = indi.id
                        self.log.info('Begin to train %s in GPU '%(file_name))
                        module_name = 'Scripts.%s'%(file_name)
                        # 从文件加载模型
                        """
                        sys.modules将自动记录该模块。当第二次再导入该模块时，python会直接到字典中查找，从而加快了程序运行的速度。
                        字典sys.modules具有字典所拥有的一切方法，可以通过这些方法了解当前的环境加载了哪些模块
                        """
                        if module_name in sys.modules.keys():
                            self.log.info('Module:%s has been loaded, delete it'%(module_name))
                            del sys.modules[module_name]
                            _module = importlib.import_module('.', module_name)
                        else:
                            _module = importlib.import_module('.', module_name)
                        # 访问对象的属性值，并提供在密钥不可用的情况下执行默认值的选项
                        _class = getattr(_module, 'RunModel')
                        cls_obj = _class()
                        # BUG 这里gpuid问题
                        p = Process(target=cls_obj.do_workk, args=(str(gpu_id), file_name,))
                        # p = Process(target=cls_obj.do_workk, args=('%d'%(gpu_id), file_name,))
                        p.start()
                        # indi.acc = Utils.get_acc_by_id(file_name)
            # 准确率达标
            else:
                file_name = indi.id
                self.log.info('%s has inherited the fitness as %.5f, no need to evaluate'%(file_name, indi.acc))
                print("writing files...")
                f = open('./populations/after_%s.txt'%(file_name[4:6]), 'a+')
                f.write('%s=%.5f\n'%(file_name, indi.acc))
                f.flush()
                f.close()



        """
        once the last individual has been pushed into the gpu, the code above will finish.
        so, a while-loop need to be insert here to check whether all GPU are available.
        Only all available are available, we can call "the evaluation for all individuals
        in this generation" has been finished.

        """
        if has_evaluated_offspring:
            all_finished = False
            while not all_finished:
                time.sleep(30)
                all_finished = gpu_status.all_gpu_available()
                # all_finished = GPUTools.all_gpu_available()
        """
        the reason that using "has_evaluated_offspring" is that:
        If all individuals are evaluated, there is no needed to wait for 300 seconds indicated in line#47
        """
        """
        When the codes run to here, it means all the individuals in this generation have been evaluated, then to save to the list with the key and value
        Before doing so, individuals that have been evaluated in this run should retrieval their fitness first.
        """
        if has_evaluated_offspring:
            file_name = './populations/after_%s.txt'%(self.individuals[0].id[4:6])
            # assert os.path.exists(file_name) # 瞎写
            f = open(file_name, 'r')
            fitness_map = {}
            for line in f:
                if len(line.strip()) > 0:
                    line = line.strip().split('=')
                    fitness_map[line[0]] = float(line[1])
            f.close()

            # todo 这里操作适应度
            # 填充缺失值0

            # 从log中加载更多数据
            # 计算新的acc 存入 chache.txt文件
            fitness_map = read_finished_acc_values(fitness_map)

            # 填充缺失值0
            fitness_map = fill_zero_values_with_average(fitness_map)

            for indi in self.individuals:
                if indi.acc == -1:
                    # 如果是 -1 说明出问题了
                    if indi.id not in fitness_map:
                        self.log.warn('The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds'%(indi.id, file_name))
                        sleep(12) #
                    indi.acc = fitness_map[indi.id]
        else:
            self.log.info('None offspring has been evaluated')

        # 保存chache.txt文件
        Utils.save_fitness_to_cache(self.individuals)






