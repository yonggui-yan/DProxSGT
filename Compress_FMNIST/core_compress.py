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
from grace_dl.dist.compressor.topk import TopKCompressor
from grace_dl.dist.compressor.none import NoneCompressor
from grace_dl.dist.compressor.qsgd import QSGDCompressor
from grace_dl.dist.memory.none import NoneMemory
 
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
          
##################################### AllReduce ##################################################
def train_AllReduce_sgd(model,epochs,train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0):
 
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID) # peers include itself
    peers = list(set(peers))  # delate the neighbor appers more than one time
    # Start optimization at the same time
     
    ws = {}
    ws_mean = {}
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
        
    WS = COMM.allgather(ws)
    with torch.no_grad():
        for name in ws.keys():
            for i in range(SIZE):
                ws_mean[name] += WS[i][name].to(device)/SIZE
      
        for name, param in model.named_parameters():
            param.copy_(ws_mean[name])
            ws[name] = param.data.detach().clone()
       
    COMM.Barrier()
    
    gamma_x = 1
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0

    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)

    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
    
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
        
        #for j in range(10):
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target); acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)
 
            optimizer.step()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
                
                WS = COMM.allgather(ws)
                for name, param in model.named_parameters():
                    ws[name] = torch.zeros_like(ws[name])
                    for i in range(SIZE):
                        ws[name] += WS[i][name].to(device)/SIZE
                         
                    param.copy_(ws[name])
                     
            COMM.Barrier()# sync at each iteration
 
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
 
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))
 
##################################### ChocoSGD ##################################################
def train_ChocoSGD(model,epochs,compressor_dx,train_loader,test_loader,optimizer,train_loader_all,gamma_x=0.8,timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0 ):

    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID) # peers include itself
    peers = list(set(peers))  # delate the neighbor appers more than one time
    # Start optimization at the same time

    ws = {}            #x
    ws_mean = {}       #
    ws_copy = {}       #x_hat
    ws_mean_copy = {}  #s
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
        ws_copy[name] = torch.zeros_like(ws[name])
        ws_mean_copy[name] = torch.zeros_like(ws[name])

    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
    #inf_norm = 0
                    
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
        
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
            
                # compress ws[name]-ws_copy[name]
                q_compressed = {}
                for name in ws.keys():
                    q_compressed_, ctx = compressor_dx.compress(tensor = ws[name]-ws_copy[name], name = name)
                    q_compressed_decompressed_ = compressor_dx.decompress(tensors = q_compressed_, ctx = ctx)
                
                    q_compressed[name] = q_compressed_
                    ws_copy[name] += q_compressed_decompressed_
                        
                for i in peers:
                    COMM.bsend(q_compressed, dest=i)
             
                for i in peers:
                    qs_received_ = COMM.recv(source = i)
                    for name in ws.keys(): 
                        if isinstance(compressor_dx, TopKCompressor):
                            ctx = ws[name].numel(), ws[name].size() # for sparsification
                        if isinstance(compressor_dx, QSGDCompressor):
                            ctx = ws[name].size() # for quantizer
                        if isinstance(compressor_dx, NoneCompressor):
                            ctx = None
                        qs_received_name_ = compressor_dx.decompress(tensors = qs_received_[name], ctx = ctx).to(device)/len(peers)
                        ws_mean_copy[name] += qs_received_name_.to(device)
   
                for name, param in model.named_parameters():
                    ws[name] = ws[name] + gamma_x*(ws_mean_copy[name]-ws_copy[name])
                    param.copy_(ws[name])
             
            COMM.Barrier()# sync at each iteration
           
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))
        
        
##################################### CDProxSGT ##################################################
def train_CDProxSGT(model,epochs,compressor_dx,compressor_gy,train_loader,test_loader,optimizer,train_loader_all,gamma_x=0.8,gamma_y=0.8, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0):

    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID) # peers include itself
    peers = list(set(peers))  # delate the neighbor appers more than one time
    # Start optimization at the same time
     
    ws = {}            #x
    ws_mean = {}       # used for consensus error
    ws_copy = {}       #x_hat
    ws_mean_copy = {}  #W x_hat
        
    yy = {}             #y
    yy_copy = {}        #y_hat
    yy_mean_copy = {}   #W y_hat
    grad_last = {}
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
        ws_copy[name] = torch.zeros_like(ws[name])
        ws_mean_copy[name] = torch.zeros_like(ws[name])
             
        yy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        yy_copy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        yy_mean_copy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        grad_last[name] = torch.zeros_like(ws[name], dtype=torch.double)

    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
    #inf_norm = 0
                    
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
        
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            for name, param in model.named_parameters():
                #inf_norm = max(inf_norm, torch.max(torch.abs(param.grad.data) ).item() )
                yy[name] += (param.grad.data.detach().clone() - grad_last[name])  ## y^{t+0.5}
                grad_last[name] = param.grad.data.detach().clone()
                    
            qy_compressed = {}
            for name in ws.keys():
                qy_compressed_, ctx = compressor_gy.compress(tensor = yy[name]-yy_copy[name], name = name)
                qy_compressed_decompressed_ = compressor_gy.decompress(tensors = qy_compressed_, ctx = ctx)
            
                qy_compressed[name] = qy_compressed_
                yy_copy[name] += qy_compressed_decompressed_
                
            for i in peers:
                COMM.bsend(qy_compressed, dest=i)
         
            for i in peers:
                yg_received_ = COMM.recv(source = i)
                for name,gard in ws.items():
                    if isinstance(compressor_gy, TopKCompressor):
                        ctx = ws[name].numel(), ws[name].size() # for sparsification
                    if isinstance(compressor_gy, QSGDCompressor):
                        ctx = ws[name].size() # for quantizer
                    if isinstance(compressor_gy, NoneCompressor):
                        ctx = None
                    yy_mean_copy[name] += compressor_gy.decompress(tensors = yg_received_[name], ctx = ctx).to(device)/len(peers)
            
            for name, param in model.named_parameters():
                yy[name] += gamma_y*(yy_mean_copy[name]-yy_copy[name])
                param.grad.copy_(yy[name].float().detach().clone()) # yy^{t+1} will used for update model
            
                #print(param.dtype,qy_compressed[name][0].dtype, yy_copy[name].dtype, yy[name].dtype)
                #return 
 
            optimizer.step()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
            
                # compress ws[name]-ws_copy[name]
                q_compressed = {}
                for name in ws.keys():
                    q_compressed_, ctx = compressor_dx.compress(tensor = ws[name]-ws_copy[name], name = name)
                    q_compressed_decompressed_ = compressor_dx.decompress(tensors = q_compressed_, ctx = ctx)
                
                    q_compressed[name] = q_compressed_
                    ws_copy[name] += q_compressed_decompressed_
                        
                for i in peers:
                    COMM.bsend(q_compressed, dest=i)
            
                for i in peers:
                    qs_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        if isinstance(compressor_dx, TopKCompressor):
                            ctx = ws[name].numel(), ws[name].size() # for sparsification
                        if isinstance(compressor_dx, QSGDCompressor):
                            ctx = ws[name].size() # for quantizer
                        if isinstance(compressor_dx, NoneCompressor):
                            ctx = None
                        ws_mean_copy[name] += compressor_dx.decompress(tensors = qs_received_[name], ctx = ctx).to(device)/len(peers)
            
                for name, param in model.named_parameters():
                    ws[name] = ws[name] + gamma_x*(ws_mean_copy[name]-ws_copy[name])
                    param.copy_(ws[name])
            
            COMM.Barrier()# sync at each iteration
           
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))
        #print('modle after epcoh', epoch, ' is saved',flush=True)
   
def train_DProxSGT(model,epochs,train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0):
 
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID)
    peers = list(set(peers))
     
    ws = {}
    ws_mean = {}
    
    grad_last = {}
        
    yy = {} #y
    yr = {} #Wy
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
                
        yy[name] = torch.zeros_like(ws[name])
        yr[name] = torch.zeros_like(ws[name])
                
        grad_last[name] = torch.zeros_like(ws[name])
        
    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    gamma_x = 1
    gamma_y = 1
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
     
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
         
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            for name, param in model.named_parameters():
                yy[name] =  yr[name]  + (param.grad.data  - grad_last[name])
                grad_last[name] = param.grad.data.detach().clone()
                
            for i in peers:
                COMM.bsend(yy, dest=i)
        
            n_peers = len(peers)
            for name in ws.keys():
                yr[name] = torch.zeros_like(ws[name])
            for i in peers:
                yy_received_ = COMM.recv(source = i)
                for name,gard in yr.items():
                    yr[name] += yy_received_[name].to(device)/len(peers)
            
            for name, param in model.named_parameters():
                param.grad.copy_(yr[name])
               
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()

                for i in peers:
                    COMM.bsend(ws, dest=i)
             
                ws_received={}
                for name in ws.keys():
                    ws_received[name] = torch.zeros_like(ws[name])
                for i in peers:
                    ws_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        ws_received[name] += ws_received_[name].to(device)/len(peers)
            
                for name in ws.keys():
                    ws[name] = ws[name] + gamma_x*(ws_received[name]-ws[name])
            
                for name, param in model.named_parameters():
                    param.copy_(ws[name])
             
            COMM.Barrier()
            
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))

# DProxCGT3 compress y only （rewrite）
def train_DProxCGT3(model,epochs,compressor_gy, gamma_y,
train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0, beta=1, maxmax=1):
 
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID)
    peers = list(set(peers))
     
    ws = {}
    ws_mean = {}
    
    grad_last = {}
        
    yy = {}             #y
    yy1 = {}
    yy_copy = {}        #y_hat
    yy_mean_copy = {}   #W y_hat
    w_qy = {}
    grad_last = {}
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name], dtype=torch.double)
                
        yy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        yy_copy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        yy_mean_copy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        
        w_qy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        grad_last[name] = torch.zeros_like(ws[name], dtype=torch.double)
        
    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    gamma_x = 1
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
    if UID==0: 
        print('epoch  UID   train_correct  \t yy_error_0 \t norm_yy_0 \t norm_yy_copy_0 \t error_yy_half \t norm_yy_copy_half \t yy_error \t norm_yy \t norm_yy1 \t test_acc \t total_time')
     
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
         
        train_stats = StatsLogger(('loss', 'correct', 'yy_error_0', 'norm_yy_0', 'norm_yy_copy_0','error_yy_half','norm_yy_copy_half', 'yy_error', 'norm_yy', 'norm_yy1'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
                        
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    yy[name] = yy[name] + (param.grad.data.detach().clone() - grad_last[name].detach().clone())  ## y^{t+0.5}
                    grad_last[name] = param.grad.data.detach().clone()
                        
                norm_yy_0 = max_fun(yy)
                norm_yy_copy_0 = max_fun(yy_copy)
                error_yy_0 = max_error(yy, yy_copy)
                
                qy_compressed = {}
                for name in ws.keys():
                    qy_compressed_ = yy[name]-yy_copy[name]
                    qy_compressed_decompressed_ = qy_compressed_.detach().clone()
                    qy_compressed[name] = qy_compressed_.detach().clone()
                    yy_copy[name] += beta*qy_compressed_decompressed_.detach().clone()


                norm_yy_copy_half = max_fun(yy_copy)
                error_yy_half = max_error(yy, yy_copy)
                           
                for i in peers:
                    COMM.bsend(qy_compressed, dest=i)
    
                for name in ws.keys():
                    w_qy[name] = torch.zeros_like(ws[name], dtype=torch.double)
                for i in peers:
                    yg_received_ = COMM.recv(source = i)
                    for name,gard in ws.items():
                        w_qy[name] +=  yg_received_[name].to(device)/len(peers)
    
                for name, param in model.named_parameters():
                    yy_mean_copy[name] += beta*w_qy[name]
                    yy[name] = yy[name] + gamma_y*(yy_mean_copy[name]-yy_copy[name])
                    
                for i in peers:
                    COMM.bsend(yy_copy, dest=i)
            
                n_peers = len(peers)
                for name in ws.keys():
                    yy1[name] = torch.zeros_like(ws[name], dtype=torch.double)#311 no
                for i in peers:
                    yy_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        yy1[name] += yy_received_[name].to(device)/len(peers)#311
                     
                #for name in ws.keys():
                #    print(qy_compressed[name].dtype, yy_copy[name].dtype, yy[name].dtype, yy1[name].dtype)
    
                error_yy = max_error(yy1, yy)
                norm_yy = max_fun(yy)
                norm_yy1 = max_fun(yy1)
                train_stats.append({'loss':float(loss),'correct':float(acc),'yy_error_0':float(error_yy_0), 'norm_yy_0':norm_yy_0, 'norm_yy_copy_0':norm_yy_copy_0,'error_yy_half':error_yy_half,'norm_yy_copy_half':norm_yy_copy_half,'yy_error':float(error_yy),'norm_yy':norm_yy,'norm_yy1':norm_yy1})

                #for name, param in model.named_parameters():
                #    param.grad.copy_(yy[name].detach().clone()) ## 498
    
                if CLIP_GRAD:
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)
    
                #optimizer.step()
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
                    ws[name] -= 0.05*yy[name].float()
                    
                for i in peers:
                    COMM.bsend(ws, dest=i)
             
                ws_received={}
                for name in ws.keys():
                    ws_received[name] = torch.zeros_like(ws[name])
                for i in peers:
                    ws_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        ws_received[name] += ws_received_[name].to(device)/len(peers)
            
                for name in ws.keys():
                    ws[name] = ws[name] + gamma_x*(ws_received[name]-ws[name])
            
                for name, param in model.named_parameters():
                    param.copy_(ws[name])
             
            COMM.Barrier()
        

        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        print_state['yy_error'] = train_stats.mean('yy_error')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
        

        print(epoch,' ',UID,'  ', int(train_stats.mean('correct')),'\t', train_stats.mean('yy_error_0'),'\t', train_stats.mean('norm_yy_0'),'\t', train_stats.mean('norm_yy_copy_0'),'\t', train_stats.mean('error_yy_half'),'\t',train_stats.mean('norm_yy_copy_half'), train_stats.mean('yy_error'),'\t', train_stats.mean('norm_yy'),'\t', train_stats.mean('norm_yy1'),'\t',int(print_state['test_acc']),'\t',int(print_state['total_time']))

         
        for logger in  loggers: logger.append(print_state)
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))



# DProxCGT2 compress y only （rewrite） directly communicate yy_copy
def train_DProxCGT2(model,epochs,compressor_gy, gamma_y,
train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0, beta=1):
 
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID)
    peers = list(set(peers))
     
    ws = {}
    ws_mean = {}
    
    grad_last = {}
        
    yy = {}             #y
    yy1 = {}
    yy_copy = {}        #y_hat
    yy_mean_copy = {}   #W y_hat
    w_qy = {}
    grad_last = {}
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
                
        yy[name] = torch.zeros_like(ws[name])
        yy_copy[name] = torch.zeros_like(ws[name])
        yy_mean_copy[name] = torch.zeros_like(ws[name])
        
        w_qy[name] = torch.zeros_like(ws[name])
        grad_last[name] = torch.zeros_like(ws[name])
        
    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    gamma_x = 1
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
     
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
         
#        train_stats = StatsLogger(('loss', 'correct', 'yy_error'))
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
            train_stats.append({'loss':float(loss),'correct':float(acc)})
                        
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    #inf_norm = max(inf_norm, torch.max(torch.abs(param.grad.data) ).item() )
                    yy[name] = yy[name] + (param.grad.data.detach().clone() - grad_last[name])  ## y^{t+0.5}
                    grad_last[name] = param.grad.data.detach().clone()
                        
#                qy_compressed = {}
                for name in ws.keys():
                    qy_compressed_, ctx = compressor_gy.compress(tensor = yy[name]-yy_copy[name], name = name)
                    qy_compressed_decompressed_ = compressor_gy.decompress(tensors = qy_compressed_, ctx = ctx)
                
#                    qy_compressed[name] = qy_compressed_
                    yy_copy[name] += beta*qy_compressed_decompressed_
 
                for i in peers:
                    COMM.bsend(yy_copy, dest=i)
            
                n_peers = len(peers)
                for name in ws.keys():
                    yy[name] = torch.zeros_like(ws[name])
                for i in peers:
                    yy_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        yy[name] += yy_received_[name].to(device)/len(peers)

                for name, param in model.named_parameters():
                    param.grad.copy_(yy[name].detach().clone())
    
                optimizer.step()

                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
                    #ws[name] -= 0.05*yy[name] 

                for i in peers:
                    COMM.bsend(ws, dest=i)
             
                ws_received={}
                for name in ws.keys():
                    ws_received[name] = torch.zeros_like(ws[name])
                for i in peers:
                    ws_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        ws_received[name] += ws_received_[name].to(device)/len(peers)
            
                for name in ws.keys():
                    ws[name] = ws[name] + gamma_x*(ws_received[name]-ws[name])
            
                for name, param in model.named_parameters():
                    param.copy_(ws[name])
             
            COMM.Barrier()
            
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        #print_state['yy_error'] = train_stats.mean('yy_error')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))

 

# DProxCGT compress y only
def train_DProxCGT(model,epochs,compressor_gy, gamma_y,
train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0):
  
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID)
    peers = list(set(peers))
     
    ws = {}
    ws_mean = {}
    
    grad_last = {}
        
    yy = {}             #y
    yy_copy = {}        #y_hat
    yy_mean_copy = {}   #W y_hat
    grad_last = {}

    w_qy = {}
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
                
        yy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        w_qy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        yy_copy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        yy_mean_copy[name] = torch.zeros_like(ws[name], dtype=torch.double)
        grad_last[name] = torch.zeros_like(ws[name], dtype=torch.double)
        
    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    gamma_x = 1
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
     
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
         
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)
 
            with torch.no_grad():
                for name, param in model.named_parameters():
                    #inf_norm = max(inf_norm, torch.max(torch.abs(param.grad.data) ).item() )
                    yy[name] = yy[name] + (param.grad.data.detach().clone() - grad_last[name])  ## y^{t+0.5}
                    grad_last[name] = param.grad.data.detach().clone()
                        
                qy_compressed = {}
                for name in ws.keys():
                    qy_compressed_, ctx = compressor_gy.compress(tensor = yy[name]-yy_copy[name], name = name)
                    qy_compressed_decompressed_ = compressor_gy.decompress(tensors = qy_compressed_, ctx = ctx)
                
                    qy_compressed[name] = qy_compressed_
                    yy_copy[name] += qy_compressed_decompressed_
      
                for i in peers:
                    COMM.bsend(qy_compressed, dest=i)
    
                for name in ws.keys():
                    w_qy[name] = torch.zeros_like(ws[name], dtype=torch.double)
                for i in peers:
                    yg_received_ = COMM.recv(source = i)
                    for name,gard in ws.items():
                        if isinstance(compressor_gy, TopKCompressor):
                            ctx = ws[name].numel(), ws[name].size() # for sparsification
                        if isinstance(compressor_gy, QSGDCompressor):
                            ctx = ws[name].size() # for quantizer
                        if isinstance(compressor_gy, NoneCompressor):
                            ctx = None
                        w_qy[name] += compressor_gy.decompress(tensors = yg_received_[name], ctx = ctx).to(device)/len(peers)
    
                for name, param in model.named_parameters():
                    yy_mean_copy[name] += w_qy[name]
                    yy[name] += gamma_y*(yy_mean_copy[name]-yy_copy[name])

                for name, param in model.named_parameters():
                    param.grad.copy_(yy[name].float().detach().clone()) ## 498
                    
                    #print(param.dtype)
                    #return 

                #for name in ws.keys():
                #    print(qy_compressed[name][0].dtype, yy_copy[name].dtype, yy[name].dtype)
                #    return 
                    

                optimizer.step()
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()

                for i in peers:
                    COMM.bsend(ws, dest=i)
             
                ws_received={}
                for name in ws.keys():
                    ws_received[name] = torch.zeros_like(ws[name])
                for i in peers:
                    ws_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        ws_received[name] += ws_received_[name].to(device)/len(peers)
            
                for name in ws.keys():
                    ws[name] = ws[name] + gamma_x*(ws_received[name]-ws[name])
            
                for name, param in model.named_parameters():
                    param.copy_(ws[name])
             
            COMM.Barrier()
            
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))
 
# choco-SGT only compress x
def train_ChocoSGT(model,epochs,compressor_dx, gamma_x, train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0):
 
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID)
    peers = list(set(peers))
     
    ws = {}            #x
    ws_mean = {}       # used for consensus error
    ws_copy = {}       #x_hat
    ws_mean_copy = {}  #W x_hat
    
    grad_last = {}
        
    yy = {} #y
    yr = {} #Wy
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
        ws_copy[name] = torch.zeros_like(ws[name])
        ws_mean_copy[name] = torch.zeros_like(ws[name])
                
        yy[name] = torch.zeros_like(ws[name])
        yr[name] = torch.zeros_like(ws[name])
                
        grad_last[name] = torch.zeros_like(ws[name])
        
    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
     
    gamma_y = 1
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
     
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
         
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            for name, param in model.named_parameters():
                yy[name] =  yr[name]  + (param.grad.data  - grad_last[name])
                grad_last[name] = param.grad.data.detach().clone()
                
            for i in peers:
                COMM.bsend(yy, dest=i)
        
            n_peers = len(peers)
            for name in ws.keys():
                yr[name] = torch.zeros_like(ws[name])
            for i in peers:
                yy_received_ = COMM.recv(source = i)
                for name,gard in yr.items():
                    yr[name] += yy_received_[name].to(device)/len(peers)
            
            for name, param in model.named_parameters():
                param.grad.copy_(yr[name])
               
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
 
                # compress ws[name]-ws_copy[name]
                q_compressed = {}
                for name in ws.keys():
                    q_compressed_, ctx = compressor_dx.compress(tensor = ws[name]-ws_copy[name], name = name)
                    q_compressed_decompressed_ = compressor_dx.decompress(tensors = q_compressed_, ctx = ctx)
                
                    q_compressed[name] = q_compressed_
                    ws_copy[name] += q_compressed_decompressed_
                        
                for i in peers:
                    COMM.bsend(q_compressed, dest=i)
            
                for i in peers:
                    qs_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        if isinstance(compressor_dx, TopKCompressor):
                            ctx = ws[name].numel(), ws[name].size() # for sparsification
                        if isinstance(compressor_dx, QSGDCompressor):
                            ctx = ws[name].size() # for quantizer
                        if isinstance(compressor_dx, NoneCompressor):
                            ctx = None
                        ws_mean_copy[name] += compressor_dx.decompress(tensors = qs_received_[name], ctx = ctx).to(device)/len(peers)
            
                for name, param in model.named_parameters():
                    ws[name] = ws[name] + gamma_x*(ws_mean_copy[name]-ws_copy[name])
                    param.copy_(ws[name])
             
            COMM.Barrier()
            
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))

##################################### old method 9, choco-CGT2 ##################################################
def train_old9(model,epochs,compressor_dx, compressor_gy, memory_y, train_loader,test_loader,optimizer,train_loader_all,gamma_x=0.8, gamma_y=0.8, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0, bound=10):

    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID) # peers include itself
    peers = list(set(peers))  # delate the neighbor appers more than one time
    # Start optimization at the same time
     
    ws = {}            #x
    ws_mean = {}       #
    ws_copy = {}       #x_hat
    ws_delta = {}      #q
    ws_mean_copy = {}  #s
        
    yy = {}             #y
    yg = {}             #g
    yg_compressed = {}
    yr = {}             #r
    grad_last = {}
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
        ws_copy[name] = torch.zeros_like(ws[name])
        ws_delta[name] = torch.zeros_like(ws[name])
        ws_mean_copy[name] = torch.zeros_like(ws[name])
             
        yy[name] = torch.zeros_like(ws[name])
        yg[name] = torch.zeros_like(ws[name])
        yg_compressed[name] = torch.zeros_like(ws[name])
        yr[name] = torch.zeros_like(ws[name])
        grad_last[name] = torch.zeros_like(ws[name])

    WS = COMM.allgather(ws)
    for name in ws.keys():
        for i in range(SIZE):
            ws_mean[name] += WS[i][name].to(device)/SIZE
       
    COMM.Barrier()
    
    print_state = {}
    print_state['UID'] =  UID
    epoch = epoch0
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
    #inf_norm = 0
                    
    while (epoch < epochs):
        epoch = epoch + 1
        print_state['epoch'] = epoch
        model.train()
        
        train_stats = StatsLogger(('loss', 'correct'))
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
 
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)

            for name, param in model.named_parameters():
             
                yy[name] = gamma_y*(yr[name]-yg[name])+yy[name] + (param.grad.data - grad_last[name])
                grad_last[name] = param.grad.data.detach().clone()
                
                param.grad.copy_(yy[name]) # yy will used for update model
                 
                yg_ = memory_y.compensate(tensor = yy[name], name = name)
                yg_compressed_, ctx = compressor_gy.compress(tensor = yg_, name = name)
                memory_y.update(tensor = yg_, name= name, compressor = compressor_gy, tensor_compressed=yg_compressed_, ctx = ctx)
                yg_compressed[name] = yg_compressed_
                yg[name] = compressor_gy.decompress(tensors = yg_compressed_, ctx = ctx)
                 
            for i in peers:
                COMM.bsend(yg_compressed, dest=i)
        
            n_peers = len(peers)
            for name in ws.keys():
                yr[name] = torch.zeros_like(ws[name])
            for i in peers:
                yg_received_ = COMM.recv(source = i)
                for name,gard in yr.items():
                    if isinstance(compressor_gy, TopKCompressor):
                        ctx = gard.numel(), gard.size() # for sparsification
                    if isinstance(compressor_gy, QSGDCompressor):
                        ctx = gard.size() # for quantizer
                    if isinstance(compressor_gy, NoneCompressor):
                        ctx = None
                    yr[name] += compressor_gy.decompress(tensors = yg_received_[name], ctx = ctx).to(device)/len(peers)
              
            optimizer.step()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
            
                for name in ws.keys():
                    ws_delta[name] = ws[name]-ws_copy[name]
                
                # compress ws_delta
                q_compressed = {}
                for name in ws.keys():
                    q_compressed_, ctx = compressor_dx.compress(tensor = ws_delta[name], name = name)
                    q_compressed_decompressed_ = compressor_dx.decompress(tensors = q_compressed_, ctx = ctx)
                
                    q_compressed[name] = q_compressed_
                    ws_copy[name] += q_compressed_decompressed_
                        
                for i in peers:
                    COMM.bsend(q_compressed, dest=i)
            
                n_peers = len(peers)
                qs_received={} # maybe do not need this variable, we can add to ws_mean_copy i.e. s directly
                for name in ws.keys():
                    qs_received[name] = torch.zeros_like(ws[name])
                for i in peers:
                    qs_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        if isinstance(compressor_dx, TopKCompressor):
                            ctx = ws[name].numel(), ws[name].size() # for sparsification
                        if isinstance(compressor_dx, QSGDCompressor):
                            ctx = ws[name].size() # for quantizer
                        if isinstance(compressor_dx, NoneCompressor):
                            ctx = None
                        qs_received[name] += compressor_dx.decompress(tensors = qs_received_[name], ctx = ctx).to(device)/len(peers)
            
                for name in ws.keys():
                    ws_mean_copy[name] += qs_received[name]
                    ws[name] = ws[name] + gamma_x*(ws_mean_copy[name]-ws_copy[name])
            
                for name, param in model.named_parameters():
                    param.copy_(ws[name])
             
            COMM.Barrier()# sync at each iteration
           
        print_state['train_time'] = timer()
        print_state['train_loss'] = train_stats.mean('loss')
        print_state['train_acc'] = train_stats.mean('correct')
        
        test(ws,print_state,model,test_loader,train_loader_all)
        
        print_state['test_time']  = timer(False);
        print_state['total_time'] = timer.total_time
         
        for logger in  loggers: logger.append(print_state)
        
        state = {'epoch':epoch, 'totaltime':timer.total_time, 'state_dict':model.state_dict()}
        torch.save(state, './checkpoints/model_'+Model_Path+'_UID'+str(UID))
        #print('modle after epcoh', epoch, ' is saved',flush=True)
   
