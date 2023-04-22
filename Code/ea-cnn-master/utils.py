import configparser
import os
import numpy as np
from subprocess import Popen, PIPE
from genetic.population import Population, Individual, DenseUnit, ResUnit, PoolUnit
import logging
import sys
import multiprocessing
import time
import os

class StatusUpdateTool(object):
    @classmethod
    def clear_config(cls):
        config = configparser.ConfigParser()
        config.read('global.ini')
        secs = config.sections()
        for sec_name in secs:
            if sec_name == 'evolution_status' or sec_name == 'gpu_running_status':
                item_list = config.options(sec_name)
                for item_name in item_list:
                    config.set(sec_name, item_name, " ")
        config.write(open('global.ini', 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        config.read('global.ini')
        config.set(section, key, value)
        config.write(open('global.ini', 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")
    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        return False
        # if rs == '1':
        #     return True
        # else:
        #     return False


    @classmethod
    def get_resnet_limit(cls):
        rs = cls.__read_ini_file('network', 'resnet_limit')
        resnet_limit = []
        for i in rs.split(','):
            resnet_limit.append(int(i))
        return resnet_limit[0], resnet_limit[1]
    @classmethod
    def get_pool_limit(cls):
        rs = cls.__read_ini_file('network', 'pool_limit')
        pool_limit = []
        for i in rs.split(','):
            pool_limit.append(int(i))
        return pool_limit[0], pool_limit[1]
    @classmethod
    def get_densenet_limit(cls):
        rs = cls.__read_ini_file('network', 'densenet_limit')
        densenet_limit = []
        for i in rs.split(','):
            densenet_limit.append(int(i))
        return densenet_limit[0], densenet_limit[1]

    @classmethod
    def get_resnet_unit_length_limit(cls):
        rs = cls.__read_ini_file('resnet_configuration', 'unit_length_limit')
        resnet_unit_length_limit = []
        for i in rs.split(','):
            resnet_unit_length_limit.append(int(i))
        return resnet_unit_length_limit[0], resnet_unit_length_limit[1]

    @classmethod
    def get_densenet_k_list(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_list')
        k_list = []
        for i in rs.split(','):
            k_list.append(int(i))
        return k_list

    @classmethod
    def get_densenet_k12(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_12')
        k12_limit = []
        for i in rs.split(','):
            k12_limit.append(int(i))
        return k12_limit[0], k12_limit[1], k12_limit[2]

    @classmethod
    def get_densenet_k20(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_20')
        k20_limit = []
        for i in rs.split(','):
            k20_limit.append(int(i))
        return k20_limit[0], k20_limit[1], k20_limit[2]

    @classmethod
    def get_densenet_k40(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_40')
        k40_limit = []
        for i in rs.split(','):
            k40_limit.append(int(i))
        return k40_limit[0], k40_limit[1], k40_limit[2]

    @classmethod
    def get_output_channel(cls):
        rs = cls.__read_ini_file('network', 'output_channel')
        channels = []
        for i in rs.split(','):
            channels.append(int(i))
        return channels
    @classmethod
    def get_input_channel(cls):
        rs = cls.__read_ini_file('network', 'input_channel')
        return int(rs)
    @classmethod
    def get_num_class(cls):
        rs = cls.__read_ini_file('network', 'num_class')
        return int(rs)
    @classmethod
    def get_input_size(cls):
        rs = cls.__read_ini_file('network', 'input_size')
        return int(rs)

    @classmethod
    def get_pop_size(cls):
        rs = cls.__read_ini_file('settings', 'pop_size')
        return int(rs)
    @classmethod
    def get_epoch_size(cls):
        rs = cls.__read_ini_file('network', 'epoch')
        return int(rs)
    @classmethod
    def get_individual_max_length(cls):
        rs = cls.__read_ini_file('network', 'max_length')
        return int(rs)

    @classmethod
    def get_genetic_probability(cls):
        rs = cls.__read_ini_file('settings', 'genetic_prob').split(',')
        p = [float(i) for i in rs]
        return p


    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_pop_size()
        params['max_len'] = cls.get_individual_max_length()
        params['image_channel'] = cls.get_input_channel()
        params['output_channel'] = cls.get_output_channel()
        params['genetic_prob'] = cls.get_genetic_probability()

        params['min_resnet'], params['max_resnet'] = cls.get_resnet_limit()
        params['min_pool'], params['max_pool'] = cls.get_pool_limit()
        params['min_densenet'], params['max_densenet'] = cls.get_densenet_limit()

        params['min_resnet_unit'], params['max_resnet_unit'] = cls.get_resnet_unit_length_limit()

        params['k_list'] = cls.get_densenet_k_list()
        params['max_k12_input_channel'], params['min_k12'], params['max_k12'] = cls.get_densenet_k12()
        params['max_k20_input_channel'], params['min_k20'], params['max_k20'] = cls.get_densenet_k20()
        params['max_k40_input_channel'], params['min_k40'], params['max_k40'] = cls.get_densenet_k40()

        return params

    @classmethod
    def get_mutation_probs_for_each(cls):
        """
        defined the particular probabilities for each type of mutation
        the mutation occurs at:
        --    add
        -- remove
        --  alter
        """
        rs = cls.__read_ini_file('settings', 'mutation_probs').split(',')
        assert len(rs) == 3
        mutation_prob_list = [float(i) for i in rs]
        return mutation_prob_list



class Log(object):
    _logger = None
    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EvoCNN")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)
    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)
class GPUTools(object):
    @classmethod
    def _get_available_gpu_plain_info(cls):
        """
        这里找到静息状态的GPU线程占用情况，然后在后面确定哪些GPU被占用了

        """
        g1="1   N/A  N/A       888      G   /usr/lib/Xorg                       4MiB"
        g2="0   N/A  N/A       90      G   /usr/bin/sddm-greeter              15MiB"
        g3="0   N/A  N/A       888      G   /usr/lib/Xorg                      11MiB"
        gpu_info_list = []
        gpu_info_list1=[]
        gpu_use_list=[]
        #read the information
        # 利用nvidia-smi获取显卡状态
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        # print(lines)
        for line_no in range(len(lines)-3, -1, -1):
            # 说明是一张卡的信息
            if lines[line_no].startswith('|==='):
                break
            else:
               gpu_info_list1.append(lines[line_no][1:-1].strip())
               gpu_info_list.append(lines[line_no][1:-1].strip())
        for i in gpu_info_list1:
            # print(i)
            if i==g1:
                # print(1)
                gpu_info_list.remove(i)
            if i==g2:
                # print(2)
                gpu_info_list.remove(i)
            if i==g3:
                # print(3)
                gpu_info_list.remove(i)
        print("Currently occupied(Use 'who' to find who is using it): ", gpu_info_list)
        #parse the information
        # 此代码只适用于两卡GPU
        if len(gpu_info_list) == 1:
            info_array = gpu_info_list[0].split(' ', 1)
            if info_array[0] == '0':
                Log.info('GPU_QUERY-GPU#1 is available, choose GPU#1')
                return 1
            else:
                Log.info('GPU_QUERY-GPU#0 is available, choose GPU#0')
                return 0
        elif len(gpu_info_list) == 0:
                # GPU outputs: No running processes found
                return 10000  # indicating all the gpus are available
        else:
            for i in range(0,len(gpu_info_list)):
                gpu_use_list.append(gpu_info_list[i].split(' ',1)[0])
            if '0' not in gpu_use_list:
                Log.info('GPU_QUERY-GPU#0 is available')
                return 0
            elif '1' not in gpu_use_list:
                Log.info('GPU_QUERY-GPU#1 is available')
                return 1
            else:
                Log.info('GPU_QUERY-No available GPU')
                return None

    @classmethod
    def _get_gpu_0_plain_info(cls):
        """
        这里找到静息状态的GPU线程占用情况，然后在后面确定哪些GPU被占用了
        bug 得定死了用哪个GPU
        """
        GPU_state_1 = 0
        GPU_state_0 = 0
        g1="1   N/A  N/A       889      G   /usr/lib/Xorg                       4MiB"
        g2="0   N/A  N/A       903      G   /usr/bin/sddm-greeter              15MiB"
        g3="0   N/A  N/A       889      G   /usr/lib/Xorg                      11MiB"

        writeList = ['/usr/lib/Xorg']
        gpu_info_list = []
        gpu_info_list1=[]
        gpu_use_list=[]
        #read the information
        # 利用nvidia-smi获取显卡状态
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        # print(lines)
        for line_no in range(len(lines)-3, -1, -1):
            # 说明是一张卡的信息
            if lines[line_no].startswith('|==='):
                break
            else:
               gpu_info_list1.append(lines[line_no][1:-1].strip())
               gpu_info_list.append(lines[line_no][1:-1].strip())
        for i in gpu_info_list1:
            for writeMember in writeList:
                if writeMember in i:
                    gpu_info_list.remove(i)

            # if i==g1:
            #
            #     gpu_info_list.remove(i)
            # if i==g2:
            #
            #     gpu_info_list.remove(i)
            # if i==g3:
            #
            #     gpu_info_list.remove(i)
        # print("Currently occupied(Use 'who' to find who is using it): ", gpu_info_list)
        #parse the information
        # 此代码只适用于单卡GPU
        for k in gpu_info_list:
            if k[0] == '1':     # 1被占用
                # Log.info('GPU_QUERY-GPU#1 is occupied')
                GPU_state_1 += 1
            elif k[0] == '0':     # 0被占用
                # Log.info('GPU_QUERY-GPU#0 is occupied')
                GPU_state_0 += 1
            else:
                pass

        if GPU_state_1 == 0:
            return 1
        # elif GPU_state_0 == 0:
        #     return 0
        else:
            return 10000
        # if len(gpu_info_list) == 1:
        #     info_array = gpu_info_list[0].split(' ', 1)
        #     if info_array[0] == '0':
        #         Log.info('GPU_QUERY-GPU#1 is available, choose GPU#1')
        #         return 1
        #     else:
        #         Log.info('GPU_QUERY-GPU#0 is available, choose GPU#0')
        #         return 0
        # elif len(gpu_info_list) == 0:
        #         # GPU outputs: No running processes found
        #         return 10000  # indicating all the gpus are available
        # else:
        #     for i in range(0,len(gpu_info_list)):
        #         gpu_use_list.append(gpu_info_list[i].split(' ',1)[0])
        #     if '0' not in gpu_use_list:
        #         Log.info('GPU_QUERY-GPU#0 is available')
        #         return 0
        #     else:
        #         Log.info('GPU_QUERY-No available GPU')
        #         return None

    @classmethod
    def all_gpu_available(cls):
        plain_info = cls._get_available_gpu_plain_info()
        if plain_info is not None and plain_info == 10000:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            return False

    @classmethod
    def detect_availabel_gpu_id(cls):
        plain_info = cls._get_available_gpu_plain_info()
        if plain_info is None:
            return None
        elif plain_info == 10000:
            print('plain_info=10000')
            Log.info('GPU_QUERY-None of the GPU is occupied, return the first one')
            Log.info('GPU_QUERY-GPU#0 is available')
            return 0
        else:
            print("return plain_info")
            return plain_info

    @classmethod
    def detect_if_gpu_0_availabel(cls):
        plain_info = cls._get_gpu_0_plain_info()
        if plain_info is None:
            return None
        elif plain_info == 10000:
            # print('plain_info=10000')
            # Log.info('All GPUs are occupied')
            return None
        else:
            # print("return plain_info")
            return plain_info
class Utils(object):
    _lock = multiprocessing.Lock()
    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock
    @classmethod
    def load_cache_data(cls):
        file_name = './populations/cache.txt'
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = '%.5f'%(float(rs_[1]))
            f.close()
        return _map

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        _map = cls.load_cache_data()
        for indi in individuals:
            _key,_str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.info('Add record into cache, id:%s, acc:%.5f'%(_key, _acc))
                f = open('./populations/cache.txt', 'a+')
                _str = '%s;%.5f;%s\n'%(_key, _acc, _str)
                f.write(_str)
                f.close()
                _map[_key] = _acc


    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        file_name = './populations/begin_%02d.txt'%(gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)
    @classmethod
    def save_population_after_crossover(cls, _str, gen_no):
        file_name = './populations/crossover_%02d.txt'%(gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = './populations/mutation_%02d.txt'%(gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)


    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        id_list = []
        for _, _, file_names in os.walk('./populations'):        #for root,dirs,files in os.walk(file):  root表示当前正在访问的文件夹路径,dirs表示该文件夹下的子目录名list,files表示该文件夹下的文件list
            for file_name in file_names:
                if file_name.startswith(prefix):     #判断文件名是否以指定字符串前缀开头   string.startswith()  string.endswith()
                    id_list.append(int(file_name[6:8]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)

    @classmethod
    def load_population(cls, prefix, gen_no):
        file_name = './populations/%s_%02d.txt'%(prefix, np.min(gen_no))
        params = StatusUpdateTool.get_init_params()
        pop = Population(params, gen_no)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(params, indi_no)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('[densenet'):
                        data_maps = line[10:-1].split(',', 5)
                        densenet_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                densenet_params['number'] = int(_value)
                            elif _key == 'amount':
                                densenet_params['amount'] = int(_value)
                            elif _key == 'k':
                                densenet_params['k'] = int(_value)
                            elif _key == 'in':
                                densenet_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                densenet_params['out_channel'] = int(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s'%( _key))
                        # get max_input_channel
                        if densenet_params['k'] == 12:
                            rs = StatusUpdateTool.get_densenet_k12()
                            densenet_params['max_input_channel'] = rs[0]
                        elif densenet_params['k'] == 20:
                            rs = StatusUpdateTool.get_densenet_k20()
                            densenet_params['max_input_channel'] = rs[0]
                        elif densenet_params['k'] == 40:
                            rs = StatusUpdateTool.get_densenet_k40()
                            densenet_params['max_input_channel'] = rs[0]
                        densenet = DenseUnit(number=densenet_params['number'], amount=densenet_params['amount'],\
                                             k=densenet_params['k'], max_input_channel=densenet_params['max_input_channel'], \
                                             in_channel=densenet_params['in_channel'], out_channel=densenet_params['out_channel'])
                        indi.units.append(densenet)
                    elif line.startswith('[resnet'):
                        data_maps = line[8:-1].split(',', 4)
                        resnet_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                resnet_params['number'] = int(_value)
                            elif _key == 'amount':
                                resnet_params['amount'] = int(_value)
                            elif _key == 'in':
                                resnet_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                resnet_params['out_channel'] = int(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s'%( _key))
                        resnet = ResUnit(number=resnet_params['number'], amount=resnet_params['amount'], \
                                         in_channel=resnet_params['in_channel'], out_channel=resnet_params['out_channel'])
                        indi.units.append(resnet)
                    elif line.startswith('[pool'):
                        pool_params = {}
                        for data_item in line[6:-1].split(','):
                            _key, _value = data_item.split(':')
                            if _key =='number':
                                indi.number_id = int(_value)
                                pool_params['number'] = int(_value)
                            elif _key == 'type':
                                pool_params['max_or_avg'] = float(_value)
                            else:
                                raise ValueError('Unknown key for load pool unit, key_name:%s'%( _key))
                        pool = PoolUnit(pool_params['number'], pool_params['max_or_avg'])
                        indi.units.append(pool)
                    else:
                        print('Unknown key for load unit type, line content:%s'%(line))
            pop.individuals.append(indi)
        f.close()

        # load the fitness to the individuals who have been evaluated, only suitable for the first generation
        if gen_no == 0:
            after_file_path = './populations/after_%02d.txt'%(gen_no)
            if os.path.exists(after_file_path):
                fitness_map = {}
                f = open(after_file_path)
                for line in f:
                    if len(line.strip()) > 0:
                        line = line.strip().split('=')
                        fitness_map[line[0]] = float(line[1])
                f.close()

                for indi in pop.individuals:
                    if indi.id in fitness_map:
                        indi.acc = fitness_map[indi.id]

        return pop

    @classmethod
    def read_template(cls):
        # _path = './template/cifar10.py'
        _path = './template/neucls.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline() #skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        #print('\n'.join(part1))

        line = f.readline().rstrip() #skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        #print('\n'.join(part2))

        line = f.readline().rstrip() #skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        #print('\n'.join(part3))
        return part1, part2, part3


    @classmethod
    def generate_pytorch_file(cls, indi):
        """"""
        # query resnet and densenet unit
        unit_list = []
        for index, u in enumerate(indi.units):
            if u.type == 1:
                layer = 'self.op%d = ResNetUnit(amount=%d, in_channel=%d, out_channel=%d)'%(index, u.amount, u.in_channel, u.out_channel)
                unit_list.append(layer)
            elif u.type == 3:
                layer = 'self.op%d = DenseNetUnit(k=%d, amount=%d, in_channel=%d, out_channel=%d, max_input_channel=%d)'%(index, u.k, u.amount, u.in_channel, u.out_channel, u.max_input_channel)
                unit_list.append(layer)
        #print('\n'.join(unit_list))

        #query fully-connect layer
        out_channel_list = []
        image_output_size = StatusUpdateTool.get_input_size()
        for u in indi.units:
            if u.type == 1:
                out_channel_list.append(u.out_channel)
            elif u.type == 3:
                out_channel_list.append(u.out_channel)
            else:
                out_channel_list.append(out_channel_list[-1])
                image_output_size = int(image_output_size/2)

        fully_layer_name = 'self.linear = nn.Linear(%d, %d)'%(image_output_size*image_output_size*out_channel_list[-1], StatusUpdateTool.get_num_class())
        #print(fully_layer_name, out_channel_list, image_output_size)

        #generate the forward part
        forward_list = []
        for i, u in enumerate(indi.units):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d'%(i-1)
            if u.type ==1:
                _str = 'out_%d = self.op%d(%s)'%(i, i, last_out_put)
                forward_list.append(_str)
            elif u.type == 3:
                _str = 'out_%d = self.op%d(%s)'%(i, i, last_out_put)
                forward_list.append(_str)
            else:
                if u.max_or_avg < 0.5:
                    _str = 'out_%d = F.max_pool2d(out_%d, 2)'%(i, i-1)
                else:
                    _str = 'out_%d = F.avg_pool2d(out_%d, 2)'%(i, i-1)
                forward_list.append(_str)
        forward_list.append('out = out_%d'%(len(indi.units)-1))
        #print('\n'.join(forward_list))

        # 读取已有模板
        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        # 使用第一部分的代码 到前向为止
        _str.extend(part1)
        _str.append('\n        %s'%('#resnet and densenet unit'))
        # 把unit list里面的元素加入进去
        for s in unit_list:
            _str.append('        %s'%(s))
        _str.append('\n        %s'%('#linear unit'))
        _str.append('        %s'%(fully_layer_name))

        _str.extend(part2)
        # forward_list 里面的网络结构元素加入进去
        for s in forward_list:
            _str.append('        %s'%(s))
        _str.extend(part3)
        #print('\n'.join(_str))
        file_name = './Scripts/%s.py'%(indi.id)
        # bug 这里写错了，没考虑到第一次运行还没有这个脚本的问题
        # 解决了，是pycharm没有上传文件
        # if os.path.isfile(file_name):
        script_file_handler = open(file_name, 'w')
        # else:
        #     script_file_handler = open(file_name, 'x')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()


if __name__ == '__main__':
    GPUTools.detect_availabel_gpu_id()






