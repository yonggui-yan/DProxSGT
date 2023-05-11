
#mpirun -np 5 python dec_FMNIST_Lenet5.py --epochs=200 --batch_size=8 --lr0=0.01 --data_divide=label --method=AllReduce  --epoch0=0 --l1=0.0001

#mpirun -np 5 python dec_FMNIST_Lenet5.py --epochs=200 --batch_size=8 --lr0=0.01 --data_divide=label --method=DProxSGT  --epoch0=0 --l1=0.0001

# Gabe-DeepStorm
#mpirun -np 5 python dec_FMNIST_Lenet5.py --epochs=200 --batch_size=8 --lr0=0.01 --data_divide=label --method=DeepSTorm  --epoch0=0 --beta=0.8 --l1=0.0001


# ProxGT-SA = DProxSGT with Large Batchsize
mpirun -np 5 python dec_FMNIST_Lenet5.py --epochs=200 --batch_size=256 --lr0=0.2  --data_divide=label --method=DProxSGT  --epoch0=0 --l1=0.0001
mpirun -np 5 python dec_FMNIST_Lenet5.py --epochs=200 --batch_size=256 --lr0=0.3 --data_divide=label --method=DProxSGT  --epoch0=0 --l1=0.0001
mpirun -np 5 python dec_FMNIST_Lenet5.py --epochs=200 --batch_size=256 --lr0=0.4 --data_divide=label --method=DProxSGT  --epoch0=0 --l1=0.0001

