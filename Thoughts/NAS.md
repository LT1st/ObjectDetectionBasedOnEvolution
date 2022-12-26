 Network Architecture Search

https://blog.csdn.net/nature553863/article/details/103825717

 # NAS Definition
- 基于搜索策略，并结合约束条件 (如accuracy、latency)，在搜索空间内 (set of candidate operations or blocks)探索最优网络结构、或组件结构 (如detector的backbone、FPN)；
- 高效的NAS算法，通常是Trade-off between data-driven and experience-driven，data-driven实现高效率、自动化搜索，experience-driven减小搜索空间、降低过拟合；
- Proxy task: 评估搜索结果 (searched architecture)