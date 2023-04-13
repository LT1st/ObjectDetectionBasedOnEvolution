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
[x] 修改默认python的运行pid
writeList 中['/usr/lib/Xorg'] 修改GPU白名单

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
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:40,717 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:40,717 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:40,718 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:45,777 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:45,777 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:45,778 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:50,837 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:50,838 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:50,838 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:36:55,897 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:36:55,898 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:36:55,898 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:00,953 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:00,954 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:00,954 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:06,007 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:06,008 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:06,008 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:11,065 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:11,066 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:11,066 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         1844MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:16,125 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:16,126 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:16,126 INFO    : All GPUs are occupied
going to record++1--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:21,187 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:21,188 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:21,188 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:26,249 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:26,249 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:26,250 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:31,312 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:31,313 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:31,314 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:36,375 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:36,376 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:36,376 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:41,435 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:41,436 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:41,436 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:46,495 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:46,495 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:46,496 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:51,551 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:51,552 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:51,552 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:37:56,610 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:37:56,610 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:37:56,611 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:01,668 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:01,669 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:01,670 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:06,725 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:06,726 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:06,727 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2976MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:11,784 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:11,785 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:11,786 INFO    : All GPUs are occupied
going to record++2--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:16,846 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:16,846 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:16,846 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:21,902 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:21,903 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:21,903 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:26,960 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:26,961 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:26,962 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:32,018 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:32,019 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:32,020 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:37,077 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:37,081 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:37,082 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:42,139 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:42,140 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:42,141 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:47,200 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:47,201 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:47,201 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:52,262 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:52,263 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:52,264 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:38:57,328 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:38:57,329 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:38:57,329 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:02,389 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:02,390 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:02,390 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:07,450 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:07,451 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:07,451 INFO    : All GPUs are occupied
going to record++3--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:12,511 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:12,511 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:12,512 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:17,569 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:17,569 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:17,570 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:22,635 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:22,636 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:22,637 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:27,694 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:27,695 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:27,696 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:32,758 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:32,758 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:32,759 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:37,819 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:37,819 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:37,820 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:42,876 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:42,877 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:42,878 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:47,937 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:47,938 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:47,939 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:52,995 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:52,996 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:52,996 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:39:58,056 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:39:58,057 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:39:58,057 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3750MiB']
2023-04-11 22:40:03,118 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:03,118 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:03,118 INFO    : All GPUs are occupied
going to record++4--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:08,179 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:08,180 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:08,180 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:13,244 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:13,245 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:13,245 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:18,304 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:18,305 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:18,306 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:23,364 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:23,365 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:23,366 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:28,423 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:28,424 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:28,424 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:33,480 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:33,480 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:33,481 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:38,541 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:38,549 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:38,550 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:43,613 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:43,614 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:43,615 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:48,677 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:48,678 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:48,679 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:53,738 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:53,739 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:53,739 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:40:58,800 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:40:58,801 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:40:58,801 INFO    : All GPUs are occupied
5--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:03,867 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:03,868 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:03,868 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:08,928 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:08,928 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:08,929 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:13,987 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:13,988 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:13,988 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:19,052 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:19,053 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:19,053 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:24,116 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:24,116 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:24,117 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:29,180 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:29,180 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:29,181 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:34,240 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:34,241 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:34,242 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:39,301 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:39,301 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:39,302 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:44,362 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:44,362 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:44,363 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:49,423 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:49,423 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:49,424 INFO    : All GPUs are occupied
going to record++6--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:54,478 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:54,479 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:54,480 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:41:59,531 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:41:59,532 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:41:59,532 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:04,591 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:04,592 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:04,592 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:09,651 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:09,651 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:09,652 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:14,702 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:14,702 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:14,702 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:19,760 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:19,761 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:19,762 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:24,822 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:24,823 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:24,824 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:29,884 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:29,885 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:29,885 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3790MiB']
2023-04-11 22:42:34,945 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:34,946 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:34,947 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2574MiB']
2023-04-11 22:42:40,000 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:40,000 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:40,009 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:42:45,064 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:45,064 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:45,065 INFO    : All GPUs are occupied
going to record++7--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:42:50,133 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:50,134 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:50,134 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:42:55,195 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:42:55,195 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:42:55,196 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:43:00,255 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:00,255 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:00,256 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:43:05,312 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:05,313 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:05,314 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:43:10,373 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:10,374 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:10,375 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:43:15,444 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:15,445 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:15,445 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:43:20,505 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:20,506 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:20,506 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:25,566 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:25,567 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:25,567 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:30,627 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:30,628 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:30,628 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:35,690 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:35,691 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:35,692 INFO    : All GPUs are occupied
going to record++8--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:40,751 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:40,752 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:40,760 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:45,820 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:45,820 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:45,821 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:50,880 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:50,881 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:50,882 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:43:55,945 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:43:55,946 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:43:55,946 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:01,006 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:01,007 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:01,007 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:06,068 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:06,069 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:06,069 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:11,129 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:11,130 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:11,131 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:16,194 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:16,194 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:16,195 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:21,253 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:21,254 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:21,255 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:26,314 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:26,315 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:26,316 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:31,380 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:31,381 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:31,382 INFO    : All GPUs are occupied
9--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:36,439 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:36,440 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:36,440 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:41,502 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:41,503 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:41,505 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:46,565 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:46,566 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:46,566 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:51,629 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:51,630 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:51,630 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:44:56,691 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:44:56,692 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:44:56,693 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:01,752 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:01,752 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:01,753 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:06,809 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:06,810 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:06,810 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:11,869 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:11,870 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:11,871 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:16,937 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:16,938 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:16,938 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:22,001 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:22,002 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:22,002 INFO    : All GPUs are occupied
going to record++10--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:27,061 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:27,062 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:27,062 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:32,121 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:32,122 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:32,122 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:37,183 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:37,184 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:37,184 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:42,240 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:42,240 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:42,241 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:47,302 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:47,303 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:47,304 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:52,371 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:52,372 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:52,372 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:45:57,432 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:45:57,433 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:45:57,434 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3768MiB']
2023-04-11 22:46:02,516 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:02,517 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:02,518 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:07,579 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:07,579 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:07,580 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:12,645 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:12,646 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:12,646 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:17,705 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:17,706 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:17,707 INFO    : All GPUs are occupied
going to record++11--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:22,764 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:22,765 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:22,766 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:27,824 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:27,824 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:27,824 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:32,881 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:32,881 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:32,882 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:37,939 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:37,940 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:37,941 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:43,001 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:43,002 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:43,002 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:48,055 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:48,056 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:48,056 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:53,114 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:53,115 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:53,116 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:46:58,175 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:46:58,176 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:46:58,176 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:03,235 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:03,236 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:03,236 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:08,292 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:08,293 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:08,293 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:13,349 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:13,350 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:13,350 INFO    : All GPUs are occupied
12--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:18,405 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:18,406 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:18,407 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:23,467 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:23,467 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:23,468 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:28,527 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:28,528 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:28,528 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:33,587 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:33,587 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:33,588 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:38,647 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:38,648 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:38,648 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:43,705 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:43,706 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:43,714 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:48,772 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:48,773 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:48,773 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:53,833 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:53,834 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:53,834 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:47:58,892 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:47:58,893 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:47:58,893 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:03,954 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:03,954 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:03,955 INFO    : All GPUs are occupied
going to record++13--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:09,013 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:09,014 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:09,014 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:14,074 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:14,075 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:14,075 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:19,136 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:19,137 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:19,138 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:24,196 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:24,197 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:24,197 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:29,259 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:29,260 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:29,261 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:34,324 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:34,324 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:34,325 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:39,379 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:39,380 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:39,381 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:44,440 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:44,440 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:44,441 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:49,513 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:49,514 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:49,514 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:54,575 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:54,576 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:54,576 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:48:59,641 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:48:59,642 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:48:59,642 INFO    : All GPUs are occupied
going to record++14--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:04,699 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:04,700 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:04,701 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:09,764 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:09,765 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:09,766 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:14,827 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:14,828 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:14,828 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:19,888 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:19,889 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:19,890 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:24,945 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:24,946 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:24,946 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:30,004 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:30,005 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:30,005 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:35,063 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:35,064 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:35,065 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:40,124 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:40,125 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:40,126 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:45,188 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:45,189 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:45,197 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:50,257 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:50,258 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:50,259 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:49:55,317 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:49:55,318 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:49:55,318 INFO    : All GPUs are occupied
15--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:00,377 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:00,378 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:00,378 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:05,438 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:05,439 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:05,439 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:10,500 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:10,501 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:10,501 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:15,566 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:15,566 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:15,567 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:20,625 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:20,626 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:20,626 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:25,683 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:25,683 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:25,684 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:30,741 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:30,742 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:30,742 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:35,800 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:35,801 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:35,802 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:40,861 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:40,861 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:40,861 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:45,920 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:45,921 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:45,921 INFO    : All GPUs are occupied
going to record++16--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:50,979 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:50,980 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:50,981 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:50:56,050 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:50:56,051 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:50:56,051 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2576MiB']
2023-04-11 22:51:01,112 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:01,113 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:01,114 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:06,177 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:06,177 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:06,177 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:11,235 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:11,236 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:11,237 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:16,288 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:16,289 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:16,289 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:21,349 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:21,350 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:21,350 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:26,410 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:26,410 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:26,410 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:31,464 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:31,464 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:31,464 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:36,521 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:36,522 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:36,522 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:51:41,578 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:41,579 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:41,580 INFO    : All GPUs are occupied
going to record++17--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:51:46,636 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:46,637 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:46,646 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:51:51,715 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:51,716 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:51,716 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:51:56,776 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:51:56,777 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:51:56,778 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:01,840 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:01,841 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:01,842 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:06,901 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:06,902 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:06,903 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:11,967 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:11,968 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:11,968 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:17,030 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:17,030 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:17,031 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:22,091 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:22,092 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:22,092 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:27,149 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:27,150 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:27,150 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:32,212 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:32,213 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:32,213 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:37,271 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:37,272 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:37,272 INFO    : All GPUs are occupied
18--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:42,336 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:42,337 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:42,337 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:47,398 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:47,398 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:47,407 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:52,470 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:52,470 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:52,471 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:52:57,534 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:52:57,535 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:52:57,535 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:02,592 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:02,593 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:02,593 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:07,653 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:07,654 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:07,654 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:12,714 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:12,715 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:12,715 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:17,773 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:17,774 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:17,775 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:22,835 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:22,836 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:22,837 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:27,900 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:27,901 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:27,902 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:32,960 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:32,961 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:32,961 INFO    : All GPUs are occupied
19--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:38,022 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:38,023 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:38,023 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:43,083 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:43,084 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:43,085 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:48,143 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:48,144 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:48,144 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:53,213 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:53,214 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:53,214 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:53:58,272 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:53:58,273 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:53:58,273 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:03,334 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:03,335 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:03,336 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:08,393 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:08,394 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:08,395 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:13,456 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:13,457 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:13,457 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:18,509 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:18,510 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:18,511 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:23,569 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:23,570 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:23,570 INFO    : All GPUs are occupied
going to record++20--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:28,626 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:28,627 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:28,628 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:33,685 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:33,686 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:33,686 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:38,746 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:38,747 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:38,747 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:43,801 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:43,802 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:43,803 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:48,857 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:48,863 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:48,864 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:53,923 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:53,924 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:53,925 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:54:58,986 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:54:58,986 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:54:58,987 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:04,046 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:04,047 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:04,048 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:09,110 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:09,111 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:09,111 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:14,169 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:14,170 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:14,171 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:19,227 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:19,228 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:19,229 INFO    : All GPUs are occupied
21--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:24,288 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:24,289 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:24,289 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:29,348 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:29,349 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:29,349 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:34,408 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:34,409 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:34,410 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:39,472 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:39,473 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:39,473 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:44,535 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:44,536 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:44,536 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:49,595 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:49,596 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:49,604 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:54,666 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:54,666 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:54,667 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:55:59,731 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:55:59,732 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:55:59,732 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:04,792 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:04,793 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:04,793 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:09,853 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:09,854 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:09,855 INFO    : All GPUs are occupied
going to record++22--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:14,912 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:14,913 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:14,913 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:19,970 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:19,970 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:19,971 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:25,031 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:25,032 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:25,032 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:30,089 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:30,090 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:30,090 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:35,161 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:35,161 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:35,161 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:40,221 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:40,221 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:40,222 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:45,279 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:45,279 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:45,280 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:50,338 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:50,339 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:50,339 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:56:55,398 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:56:55,398 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:56:55,399 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:00,457 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:00,458 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:00,459 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:05,519 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:05,520 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:05,520 INFO    : All GPUs are occupied
23--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:10,579 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:10,580 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:10,580 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:15,646 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:15,647 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:15,647 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:20,705 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:20,706 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:20,707 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:25,765 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:25,765 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:25,766 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:30,824 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:30,825 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:30,825 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:35,884 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:35,885 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:35,885 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:40,942 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:40,943 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:40,944 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:46,002 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:46,003 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:46,003 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:51,063 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:51,067 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:51,068 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:57:56,128 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:57:56,129 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:57:56,129 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:01,190 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:01,191 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:01,191 INFO    : All GPUs are occupied
24--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:06,254 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:06,254 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:06,255 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:11,319 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:11,320 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:11,321 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:16,387 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:16,388 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:16,388 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:21,449 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:21,450 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:21,450 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:26,507 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:26,508 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:26,508 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:31,567 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:31,568 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:31,568 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:36,623 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:36,624 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:36,624 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:41,680 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:41,681 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:41,681 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:46,740 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:46,741 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:46,742 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:51,802 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:51,803 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:51,811 INFO    : All GPUs are occupied
going to record++25--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:58:56,870 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:58:56,871 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:58:56,871 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:59:01,930 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:01,931 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:01,931 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:59:06,995 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:06,996 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:06,996 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:59:12,055 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:12,056 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:12,056 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3766MiB']
2023-04-11 22:59:17,115 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:17,116 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:17,116 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 22:59:22,170 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:22,171 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:22,171 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2574MiB']
2023-04-11 22:59:27,234 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:27,235 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:27,235 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:59:32,299 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:32,300 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:32,300 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:59:37,360 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:37,361 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:37,361 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:59:42,421 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:42,422 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:42,422 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:59:47,484 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:47,484 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:47,485 INFO    : All GPUs are occupied
going to record++26--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:59:52,542 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:52,543 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:52,551 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 22:59:57,612 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 22:59:57,613 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 22:59:57,614 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:00:02,674 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:02,675 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:02,675 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:00:07,731 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:07,732 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:07,732 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:12,789 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:12,790 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:12,790 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:17,848 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:17,849 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:17,850 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:22,911 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:22,911 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:22,912 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:27,972 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:27,973 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:27,974 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:33,029 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:33,030 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:33,031 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:38,094 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:38,095 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:38,095 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:43,154 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:43,155 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:43,156 INFO    : All GPUs are occupied
going to record++27--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:48,212 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:48,213 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:48,213 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:53,274 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:53,275 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:53,276 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:00:58,335 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:00:58,336 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:00:58,336 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:03,395 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:03,396 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:03,397 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:08,459 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:08,460 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:08,460 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:13,516 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:13,517 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:13,518 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:18,576 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:18,576 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:18,577 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:23,630 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:23,631 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:23,631 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:28,691 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:28,691 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:28,692 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:33,749 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:33,750 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:33,750 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:38,802 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:38,803 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:38,804 INFO    : All GPUs are occupied
going to record++28--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:43,862 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:43,863 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:43,864 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:48,924 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:48,925 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:48,926 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:53,986 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:53,987 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:53,987 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:01:59,046 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:01:59,047 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:01:59,047 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:04,101 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:04,102 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:04,102 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:09,168 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:09,169 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:09,169 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:14,229 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:14,230 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:14,231 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:19,290 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:19,291 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:19,292 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:24,349 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:24,350 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:24,350 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:29,410 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:29,411 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:29,411 INFO    : All GPUs are occupied
going to record++29--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:34,467 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:34,468 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:34,468 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:39,525 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:39,526 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:39,527 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:44,593 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:44,594 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:44,594 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:49,659 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:49,659 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:49,660 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:54,717 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:54,718 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:54,727 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:02:59,789 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:02:59,790 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:02:59,791 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:04,852 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:04,853 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:04,854 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:09,913 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:09,914 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:09,915 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:14,972 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:14,973 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:14,974 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:20,029 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:20,029 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:20,030 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:25,089 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:25,090 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:25,091 INFO    : All GPUs are occupied
going to record++30--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:30,150 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:30,150 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:30,150 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:35,210 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:35,211 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:35,212 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:40,273 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:40,274 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:40,274 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:45,337 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:45,338 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:45,338 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:50,398 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:50,399 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:50,400 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:03:55,456 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:03:55,457 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:03:55,466 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:00,530 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:00,531 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:00,532 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:05,595 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:05,596 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:05,596 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:10,655 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:10,656 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:10,657 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:15,714 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:15,715 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:15,715 INFO    : All GPUs are occupied
going to record++31--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:20,775 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:20,776 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:20,777 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:25,834 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:25,835 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:25,835 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:30,895 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:30,895 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:30,895 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:35,958 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:35,959 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:35,959 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:41,016 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:41,016 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:41,017 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:46,077 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:46,078 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:46,079 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:51,137 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:51,138 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:51,138 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:04:56,198 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:04:56,198 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:04:56,207 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:01,269 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:01,270 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:01,271 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:06,325 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:06,326 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:06,326 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:11,385 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:11,385 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:11,386 INFO    : All GPUs are occupied
32--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:16,447 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:16,448 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:16,448 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:21,503 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:21,503 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:21,504 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:26,564 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:26,564 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:26,565 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:31,623 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:31,624 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:31,624 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:36,686 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:36,687 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:36,688 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:41,749 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:41,749 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:41,750 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:46,814 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:46,815 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:46,815 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:51,877 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:51,878 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:51,879 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:05:56,936 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:05:56,937 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:05:56,938 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:01,995 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:01,996 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:01,997 INFO    : All GPUs are occupied
going to record++33--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:07,057 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:07,058 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:07,059 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:12,120 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:12,120 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:12,121 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:17,170 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:17,170 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:17,171 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:22,231 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:22,232 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:22,232 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:27,288 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:27,289 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:27,289 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:32,354 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:32,354 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:32,355 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:37,421 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:37,422 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:37,423 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:42,482 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:42,482 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:42,483 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:47,543 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:47,544 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:47,544 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:52,599 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:52,600 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:52,601 INFO    : All GPUs are occupied
going to record++34--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:06:57,657 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:06:57,658 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:06:57,666 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:02,730 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:02,731 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:02,731 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:07,793 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:07,794 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:07,794 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:12,851 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:12,852 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:12,853 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:17,909 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:17,909 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:17,910 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:22,971 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:22,972 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:22,972 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:28,029 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:28,030 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:28,031 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:33,092 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:33,093 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:33,093 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:38,161 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:38,161 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:38,161 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:43,222 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:43,223 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:43,223 INFO    : All GPUs are occupied
going to record++35--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:48,278 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:48,279 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:48,279 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:07:53,339 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:53,340 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:53,340 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2576MiB']
2023-04-11 23:07:58,399 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:07:58,399 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:07:58,407 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:03,461 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:03,462 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:03,462 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:08,524 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:08,524 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:08,525 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:13,588 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:13,589 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:13,589 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:18,650 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:18,651 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:18,652 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:23,713 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:23,713 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:23,714 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:28,773 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:28,773 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:28,774 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:33,837 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:33,838 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:33,839 INFO    : All GPUs are occupied
going to record++36--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:08:38,887 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:38,888 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:38,888 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:08:43,945 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:43,946 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:43,946 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:08:49,001 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:49,002 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:49,002 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:08:54,062 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:54,063 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:54,064 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:08:59,119 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:08:59,119 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:08:59,128 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:04,191 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:04,192 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:04,192 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:09,254 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:09,255 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:09,255 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:14,314 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:14,315 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:14,316 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:19,376 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:19,377 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:19,377 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:24,435 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:24,436 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:24,437 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:29,495 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:29,495 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:29,496 INFO    : All GPUs are occupied
37--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:34,556 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:34,556 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:34,557 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:39,620 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:39,621 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:39,622 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:44,682 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:44,683 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:44,683 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:49,743 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:49,744 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:49,745 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:54,804 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:54,805 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:54,805 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:09:59,864 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:09:59,873 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:09:59,874 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:04,935 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:04,936 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:04,936 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:09,994 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:09,994 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:09,995 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:15,051 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:15,052 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:15,052 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:20,112 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:20,113 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:20,113 INFO    : All GPUs are occupied
going to record++38--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:25,170 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:25,171 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:25,171 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:30,228 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:30,229 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:30,229 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:35,290 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:35,291 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:35,291 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:40,347 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:40,348 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:40,348 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:45,407 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:45,408 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:45,408 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:50,475 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:50,475 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:50,476 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:10:55,538 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:10:55,539 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:10:55,539 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:00,597 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:00,597 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:00,598 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:05,660 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:05,660 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:05,661 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:10,714 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:10,714 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:10,715 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:15,774 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:15,775 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:15,775 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:20,838 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:20,838 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:20,839 INFO    : All GPUs are occupied
going to record++39--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:25,893 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:25,893 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:25,894 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:30,949 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:30,950 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:30,950 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:36,005 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:36,006 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:36,007 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:41,063 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:41,064 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:41,064 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:46,125 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:46,126 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:46,126 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:51,184 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:51,185 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:51,186 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:11:56,248 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:11:56,248 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:11:56,249 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:01,310 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:01,310 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:01,311 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:06,374 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:06,375 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:06,376 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:11,438 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:11,439 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:11,439 INFO    : All GPUs are occupied
going to record++40--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:16,497 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:16,498 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:16,498 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:21,559 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:21,559 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:21,560 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:26,625 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:26,626 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:26,627 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:31,686 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:31,687 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:31,687 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:36,747 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:36,748 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:36,749 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:41,809 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:41,810 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:41,810 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:46,867 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:46,868 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:46,868 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:51,930 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:51,931 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:51,932 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:12:56,984 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:12:56,985 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:12:56,985 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:02,044 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:02,045 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:02,046 INFO    : All GPUs are occupied
41--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:07,097 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:07,097 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:07,097 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:12,155 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:12,156 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:12,156 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:17,213 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:17,214 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:17,214 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:22,270 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:22,271 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:22,271 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:27,331 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:27,332 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:27,332 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:32,389 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:32,389 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:32,389 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:37,447 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:37,448 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:37,448 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:42,507 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:42,508 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:42,509 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:47,565 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:47,566 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:47,567 INFO    : All GPUs are occupied
42--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:52,628 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:52,628 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:52,629 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:13:57,680 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:13:57,680 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:13:57,681 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:02,735 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:02,735 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:02,744 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:07,811 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:07,812 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:07,812 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:12,873 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:12,874 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:12,875 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:17,935 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:17,936 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:17,936 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:23,008 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:23,009 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:23,010 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:28,071 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:28,072 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:28,072 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:33,132 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:33,133 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:33,133 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:38,188 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:38,188 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:38,189 INFO    : All GPUs are occupied
going to record++43--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:43,245 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:43,246 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:43,246 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:48,305 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:48,306 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:48,307 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:53,362 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:53,363 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:53,363 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:14:58,417 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:14:58,418 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:14:58,418 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:03,473 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:03,474 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:03,475 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:08,534 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:08,535 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:08,536 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:13,594 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:13,594 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:13,595 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:18,654 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:18,655 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:18,655 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:23,716 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:23,717 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:23,717 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:28,774 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:28,775 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:28,776 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:33,835 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:33,836 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:33,836 INFO    : All GPUs are occupied
44--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:38,896 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:38,897 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:38,898 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:43,957 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:43,958 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:43,958 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:49,022 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:49,023 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:49,023 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:54,086 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:54,087 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:54,087 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:15:59,149 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:15:59,150 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:15:59,151 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:16:04,213 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:04,214 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:04,214 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:16:09,273 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:09,274 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:09,274 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:16:14,335 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:14,336 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:14,337 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:16:19,398 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:19,398 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:19,399 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3786MiB']
2023-04-11 23:16:24,457 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:24,458 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:24,458 INFO    : All GPUs are occupied
going to record++45--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2574MiB']
2023-04-11 23:16:29,523 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:29,524 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:29,524 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:16:34,586 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:34,586 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:34,587 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:16:39,647 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:39,648 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:39,648 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:16:44,706 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:44,707 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:44,707 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:16:49,759 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:49,760 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:49,761 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:16:54,821 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:54,822 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:54,823 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:16:59,879 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:16:59,880 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:16:59,881 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:17:04,939 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:04,940 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:04,949 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:17:10,011 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:10,012 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:10,012 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:15,075 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:15,076 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:15,076 INFO    : All GPUs are occupied
going to record++46--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:20,146 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:20,147 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:20,148 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:25,210 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:25,211 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:25,211 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:30,270 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:30,271 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:30,271 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:35,335 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:35,336 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:35,336 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:40,393 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:40,394 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:40,395 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:45,450 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:45,450 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:45,451 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:50,503 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:50,504 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:50,505 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:17:55,561 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:17:55,561 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:17:55,562 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:00,620 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:00,621 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:00,621 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:05,681 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:05,682 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:05,691 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:10,754 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:10,754 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:10,755 INFO    : All GPUs are occupied
47--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:15,816 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:15,817 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:15,817 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:20,876 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:20,877 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:20,877 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:25,936 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:25,937 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:25,937 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:30,992 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:30,993 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:30,993 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:36,052 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:36,053 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:36,053 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:41,113 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:41,114 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:41,115 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:46,177 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:46,178 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:46,179 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3768MiB']
2023-04-11 23:18:51,265 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:51,266 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:51,267 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:18:56,327 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:18:56,327 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:18:56,328 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:01,383 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:01,383 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:01,384 INFO    : All GPUs are occupied
going to record++48--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:06,441 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:06,442 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:06,443 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:11,512 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:11,513 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:11,513 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:16,573 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:16,574 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:16,575 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:21,636 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:21,637 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:21,637 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:26,699 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:26,700 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:26,700 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:31,766 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:31,767 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:31,767 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:36,828 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:36,829 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:36,829 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:41,891 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:41,892 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:41,892 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:46,949 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:46,949 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:46,950 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:52,007 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:52,008 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:52,009 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:19:57,063 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:19:57,064 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:19:57,065 INFO    : All GPUs are occupied
49--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:02,126 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:02,127 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:02,127 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:07,185 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:07,186 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:07,187 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:12,246 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:12,246 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:12,247 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:17,305 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:17,306 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:17,306 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:22,370 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:22,370 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:22,371 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:27,428 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:27,428 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:27,429 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:32,489 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:32,490 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:32,491 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:37,549 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:37,550 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:37,550 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:42,610 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:42,611 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:42,611 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:47,671 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:47,672 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:47,672 INFO    : All GPUs are occupied
going to record++50--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:52,730 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:52,731 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:52,732 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:20:57,785 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:20:57,786 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:20:57,786 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:02,846 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:02,847 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:02,848 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:07,900 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:07,901 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:07,901 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:12,958 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:12,959 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:12,959 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:18,015 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:18,016 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:18,016 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:23,074 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:23,075 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:23,076 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:28,141 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:28,142 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:28,142 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:33,204 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:33,205 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:33,206 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:38,266 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:38,267 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:38,267 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:43,323 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:43,323 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:43,324 INFO    : All GPUs are occupied
going to record++51--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:48,382 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:48,383 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:48,384 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:53,444 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:53,444 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:53,445 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:21:58,501 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:21:58,501 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:21:58,502 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:03,559 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:03,559 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:03,560 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:08,617 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:08,617 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:08,618 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:13,677 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:13,678 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:13,678 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:18,735 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:18,736 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:18,736 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:23,795 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:23,796 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:23,797 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:28,862 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:28,862 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:28,863 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:33,925 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:33,926 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:33,926 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:38,988 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:38,989 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:38,989 INFO    : All GPUs are occupied
going to record++52--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:44,047 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:44,047 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:44,048 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:49,107 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:49,108 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:49,108 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:54,162 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:54,162 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:54,163 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:22:59,222 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:22:59,222 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:22:59,223 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:04,284 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:04,284 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:04,285 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:09,350 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:09,351 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:09,351 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:14,414 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:14,415 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:14,415 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:19,471 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:19,472 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:19,472 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:24,531 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:24,532 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:24,533 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:29,590 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:29,590 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:29,591 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:34,649 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:34,650 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:34,650 INFO    : All GPUs are occupied
53--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:39,707 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:39,707 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:39,707 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:44,765 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:44,766 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:44,766 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:49,822 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:49,823 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:49,823 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:54,882 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:54,882 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:54,883 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:23:59,941 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:23:59,942 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:23:59,942 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:05,007 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:05,008 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:05,008 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:10,072 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:10,073 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:10,073 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:15,135 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:15,136 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:15,145 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:20,207 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:20,208 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:20,209 INFO    : All GPUs are occupied
going to record++54--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:25,274 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:25,274 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:25,275 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:30,330 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:30,330 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:30,331 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:35,391 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:35,392 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:35,392 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:40,454 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:40,455 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:40,455 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:45,521 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:45,522 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:45,522 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:24:50,585 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:50,585 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:50,586 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2576MiB']
2023-04-11 23:24:55,643 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:24:55,644 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:24:55,644 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:00,699 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:00,699 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:00,700 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:05,762 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:05,763 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:05,763 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:10,820 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:10,821 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:10,821 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:15,881 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:15,881 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:15,882 INFO    : All GPUs are occupied
going to record++55--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:20,936 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:20,937 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:20,938 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:25,997 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:25,998 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:25,998 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:31,059 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:31,060 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:31,060 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:25:36,126 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:36,127 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:36,127 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:25:41,190 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:41,191 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:41,191 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:25:46,246 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:46,247 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:46,247 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:25:51,306 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:51,307 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:51,307 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:25:56,364 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:25:56,364 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:25:56,365 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:01,424 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:01,425 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:01,425 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:06,486 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:06,487 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:06,487 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:11,552 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:11,553 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:11,553 INFO    : All GPUs are occupied
going to record++56--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:16,613 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:16,614 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:16,622 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:21,682 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:21,683 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:21,684 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:26,748 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:26,749 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:26,750 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:31,810 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:31,811 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:31,812 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:36,867 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:36,868 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:36,869 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:41,927 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:41,928 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:41,929 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:46,984 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:46,985 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:46,985 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:52,043 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:52,044 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:52,044 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:26:57,102 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:26:57,103 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:26:57,103 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:02,160 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:02,161 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:02,161 INFO    : All GPUs are occupied
57--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:07,221 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:07,222 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:07,222 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:12,278 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:12,278 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:12,278 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:17,336 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:17,337 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:17,338 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:22,397 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:22,398 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:22,398 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:27,458 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:27,459 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:27,459 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:32,525 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:32,525 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:32,526 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:37,585 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:37,586 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:37,586 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:42,647 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:42,647 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:42,647 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:47,703 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:47,704 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:47,704 INFO    : All GPUs are occupied
58--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:52,761 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:52,761 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:52,762 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:27:57,821 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:27:57,821 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:27:57,821 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:02,878 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:02,879 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:02,880 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:07,940 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:07,941 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:07,941 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:13,001 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:13,002 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:13,002 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:18,061 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:18,069 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:18,069 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:23,132 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:23,132 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:23,133 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:28,191 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:28,192 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:28,192 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:33,249 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:33,250 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:33,250 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:38,311 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:38,312 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:38,313 INFO    : All GPUs are occupied
going to record++59--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:43,374 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:43,375 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:43,376 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:48,439 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:48,439 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:48,440 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:53,499 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:53,500 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:53,501 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:28:58,561 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:28:58,562 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:28:58,562 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:03,622 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:03,623 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:03,624 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:08,680 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:08,681 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:08,682 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:13,741 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:13,742 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:13,742 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:18,803 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:18,804 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:18,804 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:23,861 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:23,862 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:23,863 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:28,926 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:28,927 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:28,928 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:33,989 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:33,990 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:33,990 INFO    : All GPUs are occupied
60--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:39,050 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:39,051 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:39,052 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:44,110 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:44,111 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:44,111 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:49,171 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:49,172 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:49,172 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:54,231 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:54,232 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:54,232 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:29:59,300 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:29:59,300 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:29:59,301 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:04,359 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:04,360 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:04,360 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:09,420 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:09,420 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:09,421 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:14,484 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:14,485 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:14,485 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:19,541 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:19,542 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:19,551 INFO    : All GPUs are occupied
going to record++61--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:24,613 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:24,614 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:24,614 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:29,672 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:29,673 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:29,674 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:34,735 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:34,736 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:34,736 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:39,791 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:39,792 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:39,792 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:44,855 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:44,856 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:44,857 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:49,920 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:49,921 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:49,922 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:30:54,982 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:30:54,983 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:30:54,984 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:00,041 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:00,042 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:00,043 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:05,099 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:05,100 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:05,101 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:10,162 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:10,163 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:10,164 INFO    : All GPUs are occupied
going to record++62--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:15,219 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:15,219 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:15,219 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:20,271 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:20,272 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:20,272 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:25,326 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:25,327 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:25,327 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:30,382 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:30,383 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:30,384 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:35,446 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:35,447 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:35,447 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:40,511 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:40,512 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:40,512 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:45,570 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:45,571 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:45,572 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:50,632 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:50,633 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:50,633 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:31:55,697 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:31:55,698 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:31:55,699 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:00,760 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:00,761 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:00,761 INFO    : All GPUs are occupied
going to record++63--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:05,819 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:05,820 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:05,821 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:10,877 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:10,878 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:10,879 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:15,938 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:15,939 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:15,939 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:20,995 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:20,996 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:20,996 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:26,058 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:26,059 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:26,059 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:31,123 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:31,124 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:31,124 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:36,187 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:36,187 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:36,188 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:41,247 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:41,248 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:41,249 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:46,308 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:46,309 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:46,309 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:51,364 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:51,365 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:51,365 INFO    : All GPUs are occupied
going to record++Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:32:56,425 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:32:56,426 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:32:56,426 INFO    : All GPUs are occupied
64--Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:33:01,488 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:01,489 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:01,489 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:33:06,554 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:06,555 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:06,555 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:33:11,618 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:11,619 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:11,619 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           3788MiB']
2023-04-11 23:33:16,682 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:16,683 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:16,684 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           2574MiB']
2023-04-11 23:33:21,744 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:21,744 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:21,753 INFO    : All GPUs are occupied
Currently occupied(Use 'who' to find who is using it):  ['1   N/A  N/A      1772      C   /opt/anaconda/bin/python         2978MiB', '0   N/A  N/A      1443      C   python                           4978MiB']
2023-04-11 23:33:26,813 INFO    : GPU_QUERY-GPU#1 is occupied
2023-04-11 23:33:26,813 INFO    : GPU_QUERY-GPU#0 is occupied
2023-04-11 23:33:26,814 INFO    : All GPUs are occupied
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