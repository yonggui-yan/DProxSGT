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

#loss_function = nn.CrossEntropyLoss(reduction='none')
loss_function = nn.CrossEntropyLoss()
 
CLIP_GRAD = True
CLIP_GRAD_NORM_MAX = 10

def cons_error(ws,ws_mean):
    error = 0
    for name in ws.keys():
        error += pow(torch.norm(ws[name]-ws_mean[name],'fro'),2)
    
    return error.cpu().numpy()
    
def non_zeros_percent(model):
    nnz = 0
    total_params = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            total_params += torch.numel(param.data)
            nnz += torch.sum( torch.abs(param) > 1e-6 ).item()
            
    return 100*nnz/total_params

def model_norm1(model):
    res = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            res += torch.sum( torch.abs(param) ).item()
            
    return res 
    
def prox_Norm1(model, eta=0):
    #prox_{eta|.|}(x) =  max(abs(x)-eta,0)*sign(x)
    with torch.no_grad():
        for name, param in model.named_parameters():
            ws = (param.data.detach().clone())
            param.copy_(torch.max(abs(ws)-eta,torch.zeros_like(ws))*torch.sign(ws))
            
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
                
        non0per = non_zeros_percent(model)
        norm1 = model_norm1(model)

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

        non0per_meanx = non_zeros_percent(model)
        norm1_meanx = model_norm1(model)

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
    print_state['non_zero'] = non0per
    print_state['norm1'] = norm1
    print_state['test_meanx_loss'] = test_stats_mean_x.mean('loss')
    print_state['test_meanx_acc'] = test_stats_mean_x.mean('correct')
    print_state['train_meanx_loss'] = train_all_stats_mean_x.mean('loss')
    print_state['train_meanx_acc'] = train_all_stats_mean_x.mean('correct')
    print_state['non_zero_meanx'] = non0per_meanx
    print_state['norm1_meanx'] = norm1_meanx
    
