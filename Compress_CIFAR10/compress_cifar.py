import argparse

import numpy

from mpi4py import MPI
from core_compress import *
from torch_backend import *

import models

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

parser.add_argument('--lr0', type=float, default=1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--epoch0', type=int, default=0) #25

parser.add_argument('--method', type=str, default='CDProxSGT', choices=['AllReduce', 'DProxSGT', 'CDProxSGT', 'ChocoSGD', 'old9', 'ChocoSGT', 'DProxCGT', 'DProxCGT2', 'DProxCGT3'],
                        help='solver.')
                        
parser.add_argument('--data_divide', type=str, default='label', choices=['index', 'label'], help='dividing data method')


parser.add_argument('--gamma_x', type=float, default=0.8)
parser.add_argument('--gamma_y', type=float, default=0.8)


parser.add_argument('--compress_x', type=str, default='Top40', choices=['None', 'Top30', 'Top40', 'QSGD256', 'QSGD64'], help='compressor for x')
parser.add_argument('--compress_y', type=str, default='Top40', choices=['None', 'Top30', 'Top40', 'QSGD256', 'QSGD64'], help='compressor for y')
parser.add_argument('--memory_y', type=str, default='None', choices=['None'], help='memory for y')

parser.add_argument('-a', '--arch', metavar='ARCH', default='fixup_resnet20', choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + '(default: fixup_resnet20)')

class TSVLogger:
    def __init__(self):
        if UID==0:
            self.log = ['epoch\tRank\thours\tconses_error\ttrain_loss\ttrain_acc\ttest_loss\ttest_acc \ttrain_loss_meanx\ttrain_acc_meanx\ttest_loss_meanx\ttest_acc_meanx\tyy_error']
        else:
            self.log = []

    def append(self, print_state):
        epoch= print_state['epoch']
        hours  = print_state['total_time']/ 3600
        train_loss=print_state['train_loss']; train_acc=print_state['train_acc']
        test_loss =print_state['test_loss'];  test_acc =print_state['test_acc']
        
        conses_error = print_state['conses_error']
        
        train_loss_meanx = print_state['train_meanx_loss']; train_acc_meanx = print_state['train_meanx_acc']
        test_loss_meanx  = print_state['test_meanx_loss'];  test_acc_meanx  = print_state['test_meanx_acc']
        
        if 'yy_error' in print_state.keys():
            yy_error = print_state['yy_error'];
        else:
            yy_error = 99999.99
         
        self.log.append(f'{epoch}\t{UID}\t{hours:.8f}\t{conses_error:.8f} \t{train_loss:.4f}\t{train_acc:.2f}\t{test_loss:.4f}\t{test_acc:.2f} \t{train_loss_meanx:.4f}\t{train_acc_meanx:.2f}\t{test_loss_meanx:.4f}\t{test_acc_meanx:.2f}\t{yy_error:.4f}')

    def __str__(self):
        return '\n'.join(self.log)

def get_indices(train_set,class_name):
    indices =  []
    train_set_ = []
    for i in range(len(train_set)):
        if train_set[i][1] in class_name:
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
     
    #seed = 20230102+100*UID
    seed = 20230111+100*UID
    torch.manual_seed(seed) #  set  the same initial seed, such that the initail parameters are the same.
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

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
    #print(UID,opt.get_lr())
    #return 

    dir = 'cifar10_'+str(SIZE) +'_'+ str(args.arch) +'_epochs_'+str(epochs) +'_batchsize'+str(batch_size) + '_lr0'+str(lr0) +'_DataDivide_'+str(args.data_divide)+'_method'+str(args.method)

    if args.method=='CDProxSGT' or args.method=='ChocoSGD' or args.method=='old9' or args.method=='ChocoSGT' or args.method=='DProxCGT'  or  args.method=='DProxCGT2' or  args.method=='DProxCGT3':
#        import sys
#        sys.path.append('..')
          
        from grace_dl.dist.compressor.topk import TopKCompressor
        from grace_dl.dist.compressor.qsgd import QSGDCompressor
        from grace_dl.dist.compressor.none import NoneCompressor

        if args.compress_x == 'None': compressor_dx = NoneCompressor();
        if args.compress_x == 'Top30': compressor_dx = TopKCompressor(0.3);
        if args.compress_x == 'Top40': compressor_dx = TopKCompressor(0.4);
        if args.compress_x == 'QSGD256': compressor_dx = QSGDCompressor(256);
        if args.compress_x == 'QSGD64': compressor_dx = QSGDCompressor(64);
         
        if args.compress_y == 'None': compressor_gy = NoneCompressor();
        if args.compress_y == 'Top30': compressor_gy = TopKCompressor(0.3);
        if args.compress_y == 'Top40': compressor_gy = TopKCompressor(0.4);
        if args.compress_y == 'QSGD256': compressor_gy = QSGDCompressor(256);
        if args.compress_y == 'QSGD64': compressor_gy = QSGDCompressor(64);
        if args.compress_y == 'QSGD16': compressor_gy = QSGDCompressor(16);

        if args.method == 'old9':
            if args.memory_y == 'None':
                from grace_dl.dist.memory.none import NoneMemory
                memory_y = NoneMemory()
            else:
                print('no memory')
                return
        
        #print('UID:'+str(UID))
        #print(args.compress_y)
        #print(compressor_gy)
        #return 


    TSV = TSVLogger()
    COMM.Barrier()

    if args.method=='AllReduce':
        if UID==0:
            print('train Allreduce sgd is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_AllReduce_sgd(model,epochs,train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
 
    if args.method=='ChocoSGD':
        gamma_x = args.gamma_x
        dir = dir + '_Qx' + args.compress_x + '_gamma_x'+str(gamma_x)
 
        if UID==0:
            print('train ChocoSGD is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_ChocoSGD(model,epochs,compressor_dx,train_loader,test_loader,opt,train_loader_all,gamma_x, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)

    if args.method=='CDProxSGT':
        gamma_x = args.gamma_x
        gamma_y = args.gamma_y
        dir = dir + '_Qx' + args.compress_x + '_Qy' + args.compress_y + '_gamma_x'+str(gamma_x) + '_gamma_y'+str(gamma_y)

        if UID==0:
            print('train CDProxSGT is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_CDProxSGT(model,epochs,compressor_dx,compressor_gy,train_loader,test_loader,opt,train_loader_all, gamma_x,gamma_y, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
        
    if args.method=='DProxSGT':
      
        if UID==0:
            print('train DProxSGT is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_DProxSGT(model,epochs,train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
        
    if args.method=='ChocoSGT':
        gamma_x = args.gamma_x
        dir = dir + '_Qx' + args.compress_x + '_gamma_x'+str(gamma_x)
 
        if UID==0:
            print('train ChocoSGT is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_ChocoSGT(model,epochs,compressor_dx, gamma_x,train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
     
    if args.method=='DProxCGT':
        gamma_y = args.gamma_y
        dir = dir + '_Qy' + args.compress_y  + '_gamma_y'+str(gamma_y)

        if UID==0:
            print('train DProxCGT is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_DProxCGT(model,epochs, compressor_gy, gamma_y, train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
  
    if args.method=='DProxCGT2':
        gamma_y = args.gamma_y
        dir = dir + '_Qy' + args.compress_y  + '_gamma_y'+str(gamma_y) 

        if UID==0:
            print('train DProxCGT2 is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_DProxCGT2(model,epochs, compressor_gy, gamma_y, train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
   
    if args.method=='DProxCGT3':
        gamma_y = args.gamma_y
        dir = dir + '_Qy' + args.compress_y  + '_gamma_y'+str(gamma_y) 

        if UID==0:
            print('train DProxCGT3 is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        #train_DProxCGT3(model,epochs, compressor_gy, gamma_y, train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0, beta=args.beta,maxmax=args.maxmax)
        train_DProxCGT3(model,epochs, compressor_gy, gamma_y, train_loader,test_loader,opt,train_loader_all, timer=timer,loggers=(TSV,), Model_Path=dir,epoch0=args.epoch0)
   
    if args.method=='old9':
        gamma_x = args.gamma_x
        gamma_y = args.gamma_y
        dir = dir + '_Qx' + args.compress_x + '_Qy' + args.compress_y +'_MemoryY' + args.memory_y + '_gamma_x'+str(gamma_x) + '_gamma_y'+str(gamma_y)
 
        if UID==0:
            print('train old9 is computing.....')
            print(dir)

        timer = Timer(total_time = total_time)
        train_old9(model,epochs,compressor_dx,compressor_gy,memory_y,train_loader, test_loader,opt,train_loader_all, gamma_x,gamma_y, timer=timer,loggers=(TableLogger(), TSV), Model_Path=dir,epoch0=args.epoch0)
 
    if UID == 0:
        with open('./results/logs_'+dir+'.tsv', 'w') as f:
            f.write(str(TSV)+'\n')

    COMM.Barrier()

    if UID > 0:
        with open('./results/logs_'+dir+'.tsv', 'a') as f:
            f.write(str(TSV)+'\n')

if __name__ == '__main__':
    main()
