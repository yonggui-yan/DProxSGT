#!/bin/bash
mkdir results
mkdir checkpoints

# AllReduce
mpirun -np 5 python compress_cifar.py --arch=fixup_resnet20  --epochs=500 --batch_size=64 --lr0=0.02 --data_divide=label --method=AllReduce

# DProxSGT = Non-Compressed CDProxSGT
mpirun -np 5 python compress_cifar.py --arch=fixup_resnet20  --epochs=500 --batch_size=64 --lr0=0.02 --data_divide=label --method=CDProxSGT --compress_x=None   --compress_y=None  --gamma_x=1 --gamma_y=1

# CDProxSGT
mpirun -np 5 python compress_cifar.py --arch=fixup_resnet20  --epochs=500 --batch_size=64 --lr0=0.02 --data_divide=label --method=CDProxSGT  --compress_x=Top40  --compress_y=Top40 --gamma_x=0.8  --gamma_y=0.8

# ChocoSGD
mpirun -np 5 python compress_cifar.py --arch=fixup_resnet20  --epochs=500 --batch_size=64 --lr0=0.02 --data_divide=label --method=ChocoSGD  --compress_x=Top40 --gamma_x=0.8

# Beer = CDProxSGT with a large batch size
mpirun -np 5 python compress_cifar.py --arch=fixup_resnet20  --epochs=500 --batch_size=512 --lr0=0.10 --data_divide=label --method=CDProxSGT  --compress_x=Top40  --compress_y=Top40 --gamma_x=0.8  --gamma_y=0.8
