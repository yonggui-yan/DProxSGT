import argparse
import os.path

import numpy

from mpi4py import MPI
import torch

from read_datasets import read_datasets
from solvers import *
import models
import os

import torchvision
import torchvision.transforms as transforms

model_names = sorted(name for name in models.__dict__
    if  not name.startswith("__")
    and callable(models.__dict__[name]))

COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
UID  = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_dir', type=str, default='.')

parser.add_argument('--lr0', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--epoch0', type=int, default=0) 

parser.add_argument('--method', type=str, default='DProxSGT', choices=['AllReduce', 'DProxSGT', 'DeepSTorm'],
                        help='solver.')

parser.add_argument('--data_divide', type=str, default='index', choices=['index', 'label'], help='dividing data method')

parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--l1', type=float, default=0.0001)

parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'FashionMNIST'],
                        help='solver.')

parser.add_argument('-a', '--arch', metavar='ARCH', default='fixup_resnet20', choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + '(default: fixup_resnet20)') # LeNet5 for FashionMNIST
                        
class TSVLogger:
    def __init__(self):
        if UID==0:
            self.log = ['epoch\tRank\thours\tconses_error\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc\tnon0per\tnorm1\ttrain_loss_meanx\ttrain_acc_meanx\ttest_loss_meanx\ttest_acc_meanx\tnon0per_meanx\tnorm1_meanx']
        else:
            self.log = []

    def append(self, print_state):
        epoch= print_state['epoch']
        hours  = print_state['total_time']/ 3600
        conses_error = print_state['conses_error']

        train_loss=print_state['train_loss']; train_acc=print_state['train_acc']
        test_loss =print_state['test_loss'];  test_acc =print_state['test_acc'] 
        non0per = print_state['non_zero']
        norm1 =  print_state['norm1']
        
        train_loss_meanx = print_state['train_meanx_loss']; train_acc_meanx = print_state['train_meanx_acc']
        test_loss_meanx  = print_state['test_meanx_loss'];  test_acc_meanx  = print_state['test_meanx_acc']
        non0per_meanx = print_state['non_zero_meanx']
        norm1_meanx =  print_state['norm1_meanx']

        self.log.append(f'{epoch}\t{UID}\t{hours:.8f}\t{conses_error:10.8f} \t{train_loss:8.4f}\t{train_acc:8.2f}\t{test_loss:8.4f}\t{test_acc:8.2f} \t{non0per:8.4f}\t{norm1:8.4f}\t{train_loss_meanx:8.4f}\t{train_acc_meanx:8.2f}\t{test_loss_meanx:8.4f}\t{test_acc_meanx:8.2f}\t{non0per_meanx:8.4f}\t{norm1_meanx:8.4f}')

    def __str__(self):
        return '\n'.join(self.log)


def get_indices(train_set,class_name):
    indices =  []
    train_set_ = []
    for i in range(len(train_set)):
        if train_set[i][1] in class_name:
        #if train_set[i][1] == class_name:
            indices.append(i)
            train_set_.append(train_set[i])
    return indices, train_set_

def main():


    args = parser.parse_args()
  
    lr0 = args.lr0;
    batch_size = args.batch_size;
    epochs = args.epochs
    epoch0 = args.epoch0
    total_time = 0
     
    seed = 20210311+100*UID
    torch.manual_seed(seed) #  set  the same initial seed, such that the initail parameters are the same.
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    
    # Data
    train_dataset, test_dataset = read_datasets(args.dataset, args.data_dir, device=device)
    if train_dataset == None and test_dataset == None:
        return
     
#    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
     
    if args.data_divide=='label':# divide by the label
        idx,_  = get_indices(train_dataset, [UID,9-UID])
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idx)
        
    if args.data_divide=='index':# divide by the index, shuffle within each worker, but not between workers
        num_per_node = int(len(train_dataset)/SIZE)
        idx = [num_per_node*UID + i for i in range(num_per_node)]
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, sampler=sampler,drop_last=True)
    
    train_loader_all = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    test_loader      = torch.utils.data.DataLoader(test_dataset,  batch_size=1000, shuffle=False)
  
    # Model
    model = models.__dict__[args.arch]().to(device)
  
    base_learning_rate = lr0
    lr_schedule = lambda epoch: 1*(epoch<=epochs) #LR_1003

    lr = lambda step: base_learning_rate*lr_schedule(step // len(train_loader))
    opt = SGD(trainable_params(model), lr=lr, weight_decay= args.weight_decay, step_number=epoch0*len(train_loader))
      
    dir = args.dataset + '_Size'+str(SIZE) +'_'+ str(args.arch) +'_epochs_'+str(epochs) +'_batchsize'+str(batch_size) + '_lr0'+str(lr0) +'_data_divide'+str(args.data_divide)+ 'l1' + str(args.l1)+'_method'+str(args.method)
 
    TSV = TSVLogger()
    COMM.Barrier()
    
    if args.method=='AllReduce':
        print('train_allreduce sgd is computing.....')
        if UID==0: print(dir) 
        timer = Timer(total_time = total_time)
        train_AllReduce_sgd(model,epochs,train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0,l1=args.l1)
 
    if args.method=='DProxSGT':
        print('train_DProxSGT is computing.....')
        if UID==0: print(dir) 
        timer = Timer(total_time = total_time)
        train_DProxSGT(model,epochs,train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0,l1=args.l1)
 
    if args.method=='DeepSTorm':
        print('train_DeepSTorm is computing.....')
        dir = dir + '_beta_' + str(args.beta)
        if UID==0: print(dir) 
        timer = Timer(total_time = total_time)
        train_DeepSTorm(model,epochs,train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0, beta=args.beta, l1=args.l1)
 
    if UID == 0:
        with open('./results/logs_'+dir+'.tsv', 'w') as f:
            f.write(str(TSV)+'\n')

    COMM.Barrier()

    if UID > 0:
        with open('./results/logs_'+dir+'.tsv', 'a') as f:
            f.write(str(TSV)+'\n') 

if __name__ == '__main__':
    main()
