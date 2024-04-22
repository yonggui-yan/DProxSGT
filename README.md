# DProxSGT

This repository is the implementations of the paper: Compressed Decentralized Proximal Stochastic Gradient Method for Nonconvex Composite Problems with Heterogeneous Data [(Yan et al. 2023)](#DProxSGT)

- RegL1_FMNIST: code for training a sparse neural network model (LeNet5) on Fashion-MNIST with L1-regularlization using non-compressed methods
- RegL1_CIFAR10: code for training a sparse neural network model (FixupResnet20) on CIFAR10 with L1-regularlization using non-compressed methods

- Compress_FMNIST: code for training a neural network model (LeNet5) on Fashion-MNIST using compressed methods
- Compress_CIFAR10:  code for training a neural network model (FixupResnet20) on CIFAR10 using compressed methods

In each directory, executing the file run.sh will yield the results

## Reference  

- <a name="DProxSGT"></a>Yonggui Yan, Jie Chen, Pin-Yu Chen, Xiaodong Cui, Songtao Lu, and Yangyang Xu. [Compressed Decentralized Proximal Stochastic Gradient Method for Nonconvex Composite Problems with Heterogeneous Data](https://proceedings.mlr.press/v202/yan23a/yan23a.pdf). In International Conference on Machine Learning, pp. 39035-39061. PMLR, 2023.
