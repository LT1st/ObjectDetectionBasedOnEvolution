# BUG
```
/opt/anaconda/bin/python /tmp/pycharm_project_3/evolve.py 
2023-04-07 21:52:31,165 INFO    : *************************
Using existing mid data? False
2023-04-07 21:52:31,166 INFO    : Initialize...
2023-04-07 21:52:31,169 INFO    : EVOLVE[0-gen]-Begin to evaluate the fitness
2023-04-07 21:52:31,169 INFO    : Begin to generate python files
2023-04-07 21:52:31,181 INFO    : Finish the generation of python files
2023-04-07 21:52:31,181 INFO    : Query fitness from cache
2023-04-07 21:52:31,181 INFO    : Total hit 0 individuals for fitness
Currently occupied(Use 'who' to find who is using it):  ['0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 21:53:31,287 INFO    : GPU_QUERY-GPU#1 is available, choose GPU#1
return plain_info
2023-04-07 21:53:31,288 INFO    : Begin to train indi0000
Files already downloaded and verified
Files already downloaded and verified
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1106MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 21:54:32,562 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1106MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 21:59:32,720 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:04:32,879 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:09:32,988 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:14:33,145 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:19:33,295 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:24:33,400 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:29:33,557 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:34:33,643 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:39:33,809 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:44:33,913 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:49:33,973 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:54:34,132 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 22:59:34,292 INFO    : GPU_QUERY-No available GPU
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A   2522618      C   /opt/anaconda/bin/python         1124MiB', '0   N/A  N/A    274530      C   python                            712MiB']
2023-04-07 23:04:34,443 INFO    : GPU_QUERY-No available GPU

Process finished with exit code -1

```

单次尝试bug
```
/opt/anaconda/bin/python /tmp/pycharm_project_663/genetic/single_run.py 
Currently occupied(Use 'who' to find who is using it):  []
plain_info=10000
2023-04-11 11:34:02,984 INFO    : GPU_QUERY-None of the GPU is occupied, return the first one
2023-04-11 11:34:02,984 INFO    : GPU_QUERY-GPU#0 is available
Process Process-1:
Traceback (most recent call last):
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 1346, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/opt/anaconda/lib/python3.9/http/client.py", line 1285, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/opt/anaconda/lib/python3.9/http/client.py", line 1331, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/opt/anaconda/lib/python3.9/http/client.py", line 1280, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/opt/anaconda/lib/python3.9/http/client.py", line 1040, in _send_output
    self.send(msg)
  File "/opt/anaconda/lib/python3.9/http/client.py", line 980, in send
    self.connect()
  File "/opt/anaconda/lib/python3.9/http/client.py", line 1447, in connect
    super().connect()
  File "/opt/anaconda/lib/python3.9/http/client.py", line 946, in connect
    self.sock = self._create_connection(
  File "/opt/anaconda/lib/python3.9/socket.py", line 844, in create_connection
    raise err
  File "/opt/anaconda/lib/python3.9/socket.py", line 832, in create_connection
    sock.connect(sa)
TimeoutError: [Errno 110] Connection timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/anaconda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/opt/anaconda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/tmp/pycharm_project_663/Scripts/indi0000.py", line 220, in do_workk
    m = TrainModel()
  File "/tmp/pycharm_project_663/Scripts/indi0000.py", line 135, in __init__
    trainloader, validate_loader = data_loader.get_train_valid_loader('./data', batch_size=8, augment=True, valid_size=0.1, shuffle=True, random_seed=2312390, show_sample=False, num_workers=1, pin_memory=True)
  File "/tmp/pycharm_project_663/data_loader.py", line 76, in get_train_valid_loader
    train_dataset = datasets.CIFAR10(
  File "/opt/anaconda/lib/python3.9/site-packages/torchvision/datasets/cifar.py", line 65, in __init__
    self.download()
  File "/opt/anaconda/lib/python3.9/site-packages/torchvision/datasets/cifar.py", line 139, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/opt/anaconda/lib/python3.9/site-packages/torchvision/datasets/utils.py", line 447, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/opt/anaconda/lib/python3.9/site-packages/torchvision/datasets/utils.py", line 147, in download_url
    url = _get_redirect_url(url, max_hops=max_redirect_hops)
  File "/opt/anaconda/lib/python3.9/site-packages/torchvision/datasets/utils.py", line 95, in _get_redirect_url
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 214, in urlopen
    return opener.open(url, data, timeout)
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 517, in open
    response = self._open(req, data)
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 534, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 494, in _call_chain
    result = func(*args)
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 1389, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "/opt/anaconda/lib/python3.9/urllib/request.py", line 1349, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>

Process finished with exit code 0

```
数据集问题
- 没权限
- 没联网
- 指定目录失效

多卡问题
```
2023-04-11 19:35:25,512 INFO    : *************************
Using existing mid data? False
2023-04-11 19:35:25,513 INFO    : Initialize...
2023-04-11 19:35:25,516 INFO    : EVOLVE[0-gen]-Begin to evaluate the fitness
2023-04-11 19:35:25,516 INFO    : Begin to generate python files in evaluate.py line:16
2023-04-11 19:35:25,530 INFO    : Finish the generation of python files
2023-04-11 19:35:25,530 INFO    : Try to query fitness from cache from ./populations/cache.txt
2023-04-11 19:35:25,530 INFO    : Total hit 0 individuals for fitness
Currently occupied(Use 'who' to find who is using it):  []
plain_info=10000
2023-04-11 19:36:25,644 INFO    : GPU_QUERY-None of the GPU is occupied, return the first one
2023-04-11 19:36:25,644 INFO    : GPU_QUERY-GPU#0 is available
2023-04-11 19:36:25,645 INFO    : Begin to train indi0000
load data from: ./data ['cifar-10-batches-py.zip', 'cifar-10-batches-py']
Currently occupied(Use 'who' to find who is using it):  ['0   N/A  N/A      2374      C   /opt/anaconda/bin/python         2812MiB']
2023-04-11 19:37:26,904 INFO    : GPU_QUERY-GPU#1 is available, choose GPU#1
return plain_info
2023-04-11 19:37:26,905 INFO    : Begin to train indi0001
load data from: ./data ['cifar-10-batches-py.zip', 'cifar-10-batches-py']
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      2504      C   /opt/anaconda/bin/python         2820MiB', '0   N/A  N/A      2374      C   /opt/anaconda/bin/python         2812MiB']
2023-04-11 19:38:27,044 INFO    : GPU_QUERY-No available GPU

Process finished with exit code -1
```
只有第一次迭代可以打印出来？
网络问题？

```
/opt/anaconda/bin/python /home/lutao/Code/ea-cnn-master/evolve.py 
2023-04-11 22:36:04,039 INFO    : *************************
Using existing mid data? False
2023-04-11 22:36:04,040 INFO    : Initialize population...
2023-04-11 22:36:04,043 INFO    : EVOLVE[0-gen]-Begin to evaluate the fitness
2023-04-11 22:36:04,043 INFO    : Begin to generate python files in evaluate.py line:16
2023-04-11 22:36:04,059 INFO    : Finish the generation of python files
2023-04-11 22:36:04,059 INFO    : Try to query fitness from cache from ./populations/cache.txt
2023-04-11 22:36:04,059 INFO    : Total hit 0 individuals for fitness
Currently occupied(Use 'who' to find who is using it):  ['0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:14,115 INFO    : GPU_QUERY-GPU#0 is occupied
return plain_info
2023-04-11 22:36:14,116 INFO    : Begin to train indi0000
do_workk**load data from: ./data ['cifar-10-batches-py.zip', 'cifar-10-batches-py']
0--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:25,535 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:25,536 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:25,537 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:30,596 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:30,597 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:30,598 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:35,657 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:35,658 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:35,658 INFO    : All GPUs are occupied
2023-04-11 22:41:34,240 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:34,241 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:34,242 INFO    : All GPUs are occupied
Currently occupied(Use '
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:33:31,871 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:31,872 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:31,872 INFO    : All GPUs are occupied

Process finished with exit code -1

```
找到原因了，是因为桌面和服务器的网络连接有问题，因为程序是在桌面打开，通过网络连接到服务器，在服务器上运行。网络会间歇性地断开连接，导致程序中止。
https://stackoverflow.com/questions/75077752/pycharm-error-process-finished-with-exit-code-1

服务器跑sh脚本

注意超出显存不会报错


## 切换数据集
epoch出问题
```
epoch: 5
epoch: 0
Exception occurs, file:indi0000, pid:18435...list index out of range
finally

Process finished with exit code 0

```
类别应该编码才对
```
Traceback (most recent call last):
  File "/opt/anaconda/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3457, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-3-bc58608df5a3>", line 1, in <module>
    for _, data in enumerate(self.trainloader, 0):
  File "/opt/anaconda/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 628, in __next__
    data = self._next_data()
  File "/opt/anaconda/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 671, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/opt/anaconda/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 58, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/opt/anaconda/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py", line 58, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/lutao/Code/ea-cnn-master/NEU_CLS.py", line 118, in __getitem__
    return image, torch.tensor(int(label))
ValueError: invalid literal for int() with base 10: 'Scratches'
```
??跑不出一个额【och
```
0Exception occurs, file:indi0000, pid:1387...list index out of range
```

BUG: indi.acc 是 -1
utils.py 430 行

BUG : ACC越来越低
可能是dataloader写错了
或者是没有shuffle
```

```

BUG ： 意外退出可能是Loss NAN
```
0--going to record++Train-Epoch:  1,  Loss: 20.763, Acc:0.902
1--going to record++Train-Epoch:  2,  Loss: nan, Acc:0.296
2--going to record++Train-Epoch:  3,  Loss: nan, Acc:0.152
3--going to record++Train-Epoch:  4,  Loss: nan, Acc:0.152
4--going to record++Train-Epoch:  5,  Loss: nan, Acc:0.152
5--going to record++Train-Epoch:  6,  Loss: nan, Acc:0.152
6--going to record++Train-Epoch:  7,  Loss: nan, Acc:0.152
7--going to record++Train-Epoch:  8,  Loss: nan, Acc:0.152
8--going to record++Train-Epoch:  9,  Loss: nan, Acc:0.152
9--going to record++Train-Epoch: 10,  Loss: nan, Acc:0.152
```

Swin系列 -1 退出
- 爆显存
- Loss太大了  if epoch < 25: lr = 0.001


问题原因 设置为了200size的6类，而生成出来的网络用于10类分类
```
0--Exception occurs, file:indi0002, pid:9520...mat1 and mat2 shapes cannot be multiplied (8x18432 and 512x10)
72*256
0Exception occurs, file:indi0000, pid:14192...mat1 and mat2 shapes cannot be multiplied (10x18432 and 512x6)
finally

```

-- 发现问题 --
为什么bs调不高，是因为前向时候新建了很多层out
内存占用增加：每次创建新变量都会占用内存空间，如果在前向传播过程中有大量的临时变量创建，
可能会导致内存占用增加，特别是在处理大规模数据或者复杂模型时。

 torch.float32
 [10, 256, 200, 200] = 409600000字节 =  3.056640625 GB
 
# 修改后还是有bug
torch.cuda.empty_cache()