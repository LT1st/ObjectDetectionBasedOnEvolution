from utils import Utils,GPUTools
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep
from utils import StatusUpdateTool, Utils, Log
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
import numpy as np
import copy


# gpu_id = GPUTools.detect_availabel_gpu_id()
gpu_id = 1
file_name = 'indi0000.py'
# self.log.info('Begin to train %s' % (file_name))
# module_name = 'Scripts.%s' % (file_name)
module_name = './indi0000.py'
# 从文件加载模型
if module_name in sys.modules.keys():
    # self.log.info('Module:%s has been loaded, delete it' % (module_name))
    del sys.modules[module_name]
    _module = importlib.import_module('.', module_name)
else:
    _module = importlib.import_module('.', module_name)
# 访问对象的属性值，并提供在密钥不可用的情况下执行默认值的选项
_class = getattr(_module, 'RunModel')
cls_obj = _class()
p = Process(target=cls_obj.do_workk, args=('%d' % (gpu_id), file_name,))
p.start()