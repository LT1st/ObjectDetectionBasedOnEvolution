# ae-cnn
This is the code for the paper of "Completely Automated CNN Architecture Design Based on Blocks" published by TNNLS.
Very much apprciate if you could cite this paper when you get help from this code.

Yanan Sun, Bing Xue, Mengjie Zhang, Gary G. Yen, “Completely automated CNN architecture design based on blocks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 4, pp. 1242-1254, 2020. 


@article{sun2019completely,  
  title={Completely automated CNN architecture design based on blocks},  
  author={Sun, Yanan and Xue, Bing and Zhang, Mengjie and Yen, Gary G},  
  journal={IEEE transactions on neural networks and learning systems},  
  volume={31},  
  number={4},  
  pages={1242--1254},  
  year={2019},  
  publisher={IEEE}  
}

# Usage 
维护一系列.py文件

## utils.py
_get_available_gpu_plain_info()
修改默认python的运行pid

## evaluate.py
releaseGPUTime 调整等待时间
rGPUTime

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