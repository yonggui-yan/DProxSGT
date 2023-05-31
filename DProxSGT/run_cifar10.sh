# AllReduce
mpirun -np 5 python dec_main.py --dataset=CIFAR10 --arch=fixup_resnet20 --epochs=500 --batch_size=64 --lr0=0.02  --data_divide=label --method=AllReduce  --epoch0=0 --l1=0.00005

# DProxSGT
mpirun -np 5 python dec_main.py --dataset=CIFAR10 --arch=fixup_resnet20 --epochs=500 --batch_size=64 --lr0=0.02 --data_divide=label --method=DProxSGT  --epoch0=0 --l1=0.00005

# DeepStorm
mpirun -np 5 python dec_main.py --dataset=CIFAR10 --arch=fixup_resnet20 --epochs=500 --batch_size=64 --lr0=0.02 --data_divide=label --method=DeepSTorm  --epoch0=0 --beta=0.8 --l1=0.00005

# ProxGT-SA = DProxSGT with Large Batchsize
mpirun -np 5 python dec_main.py --dataset=CIFAR10 --arch=fixup_resnet20 --epochs=500 --batch_size=512 --lr0=0.1  --data_divide=label --method=DProxSGT  --epoch0=0 --l1=0.00005
 
