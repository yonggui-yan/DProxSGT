import torch

import torch.nn as nn
import numpy as np
from mpi4py import MPI

from torch_backend import *

COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
UID  = MPI.COMM_WORLD.Get_rank()

BUFF_SISE = 24000000 + MPI.BSEND_OVERHEAD
buff = np.empty(5*BUFF_SISE, dtype='b')
MPI.Attach_buffer(buff)

import sys
sys.path.append('..')
 
#loss_function = nn.CrossEntropyLoss(reduction='none')
loss_function = nn.CrossEntropyLoss()
 
CLIP_GRAD = True
CLIP_GRAD_NORM_MAX = 10

def norm_fun(ws):
    norm = 0
    for name in ws.keys():
        norm += pow(torch.norm(ws[name],'fro'),2)
                        
    return norm.cpu().numpy()

def cons_error(ws,ws_mean):
    error = 0
    for name in ws.keys():
        error += pow(torch.norm(ws[name]-ws_mean[name],'fro'),2)
    
    return error.cpu().numpy()
    
def max_fun(ws):
    max0 = 0
    for name in ws.keys():
        max0 = np.max([torch.max(torch.abs(ws[name])).cpu().item(), max0])
    
    return max0

def max_error(ws,ws_mean):
    max0 = 0
    for name in ws.keys():
        max0 = np.max([torch.max(torch.abs(ws[name]-ws_mean[name])).cpu().item(), max0])
    
    return max0
    
def test(ws,print_state,model,test_loader,train_loader_all):
    WS = COMM.allgather(ws)
    ws_mean = {}
    for name in ws.keys():
        ws_mean[name] = torch.zeros_like(ws[name])
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
         
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch] = conses_error

    print_state['conses_error'] = conses_error
    
    model.train(False)
    with torch.no_grad():
        
        test_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
            test_stats.append({'loss':float(loss),'correct':float(acc)})
         
        for name, param in model.named_parameters():
            param.copy_(ws_mean[name])
        test_stats_mean_x = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
            test_stats_mean_x.append({'loss':float(loss),'correct':float(acc)})

        train_all_stats_mean_x = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader_all):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
            train_all_stats_mean_x.append({'loss':float(loss),'correct':float(acc)})

        for name, param in model.named_parameters():
            param.copy_(ws[name])
                                                         
    print_state['test_loss'] = test_stats.mean('loss')
    print_state['test_acc'] = test_stats.mean('correct')
    print_state['test_meanx_loss'] = test_stats_mean_x.mean('loss')
    print_state['test_meanx_acc'] = test_stats_mean_x.mean('correct')
    print_state['train_meanx_loss'] = train_all_stats_mean_x.mean('loss')
    print_state['train_meanx_acc'] = train_all_stats_mean_x.mean('correct')
          
