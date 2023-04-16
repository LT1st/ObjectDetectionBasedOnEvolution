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

## spliteRawNEUCLS.py
分割数据集

# todo
## 修改到可以使用
- [ ] 数据加载器
- 训练文件neucls.py
- 