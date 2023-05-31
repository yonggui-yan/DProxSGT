# AllReduce
mpirun -np 5 python compress_FMNIST_Lenet5.py  --epochs=100 --batch_size=8 --lr0=0.01 --data_divide=label --method=AllReduce

# DProxSGT
mpirun -np 5 python compress_FMNIST_Lenet5.py  --epochs=100 --batch_size=8 --lr0=0.01 --data_divide=label --method=DProxSGT

# CDProxSGT
mpirun -np 5 python compress_FMNIST_Lenet5.py  --epochs=100 --batch_size=8 --lr0=0.01 --data_divide=label --method=CDProxSGT  --compress_x=Top30  --compress_y=Top30 --gamma_x=0.5  --gamma_y=0.5

# Choco-SGD
mpirun -np 5 python compress_FMNIST_Lenet5.py  --epochs=100 --batch_size=8 --lr0=0.01 --data_divide=label --method=ChocoSGD  --compress_x=Top30 --gamma_x=0.5

# Beer = CDProxSGT with a large batch size
mpirun -np 5 python compress_FMNIST_Lenet5.py  --epochs=100 --batch_size=256 --lr0=0.3 --data_divide=label --method=CDProxSGT  --compress_x=Top30  --compress_y=Top30 --gamma_x=0.5  --gamma_y=0.5
