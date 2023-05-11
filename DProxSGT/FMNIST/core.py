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
        
##################################### method 3 ##################################################
def train_AllReduce_sgd(model,epochs,train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0, l1=0.0001):
 
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
    
    #Conses_Errors = np.zeros(epochs+1)
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    #Conses_Errors[epoch]= conses_error
    
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
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
                        
            train_stats.append({'loss':float(loss),'correct':float(acc)})
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)
 
            optimizer.step()
            prox_Norm1(model, l1*optimizer.get_lr())

            with torch.no_grad():

                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()
 
                WS = COMM.allgather(ws)
                for name in ws.keys():
                    ws[name] = torch.zeros_like(ws[name])
                    for i in range(SIZE):
                        ws[name] += WS[i][name].to(device)/SIZE
                        
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
 
def train_DProxSGT(model,epochs,train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0, l1=0.0001):
 
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
                param.grad.copy_(yy[name]) # yy will used for update model
                
            for i in peers:
                COMM.bsend(yy, dest=i)
        
            n_peers = len(peers)
            for name in ws.keys():
                yr[name] = torch.zeros_like(ws[name])
            for i in peers:
                yy_received_ = COMM.recv(source = i)
                for name,gard in yr.items():
                    yr[name] += yy_received_[name].to(device)/len(peers)
              
            optimizer.step()
            prox_Norm1(model, l1*optimizer.get_lr())

            with torch.no_grad():
                for name, param in model.named_parameters():
                    ws[name] = param.data.detach().clone()

                for i in peers:
                    COMM.bsend(ws, dest=i)
            
                n_peers = len(peers)
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
 
 
def train_DeepSTorm(model,epochs,train_loader,test_loader,optimizer,train_loader_all, timer=Timer(), loggers=(TableLogger()),Model_Path = 'model_path', epoch0=0, beta=0.5, l1=0.0001):
 
    peers = [(UID + 1) % SIZE, (UID - 1) % SIZE] # the left and right  neighbor
    
    peers.append(UID)
    peers = list(set(peers))
     
    ws = {}
    ws_mean = {}
    
    dd = {} #d
    uu = {} #u
    yy = {} #y
    for name, param in model.named_parameters():
        ws[name] = param.data.detach().clone()
        ws_mean[name] = torch.zeros_like(ws[name])
                                
        dd[name] = torch.zeros_like(ws[name])
        uu[name] = torch.zeros_like(ws[name])
        yy[name] = torch.zeros_like(ws[name])
        
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
     
    error = cons_error(ws,ws_mean)
    conses_error = COMM.allreduce(error, op=MPI.SUM)
    
    print('UID',UID,'epoch',epoch,'_conses error=',conses_error,flush=True)
    
    ### initialize
    initial_batchsize = 200
    num_batch = initial_batchsize/8 # batchsize=8
    for batch_index, (data, target) in enumerate(train_loader): # use 1000 as the initial batchsize
        if batch_index >= num_batch:
            break
            
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            dd[name] += param.grad.data.detach().clone()/num_batch
           
    for i in peers:
        COMM.bsend(dd, dest=i)
         
    for i in peers:
        yy_received_ = COMM.recv(source = i)
        for name in yy.keys():
            yy[name] += yy_received_[name].to(device)/len(peers)
     
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
            loss.backward() # calculate u
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)
       
            with torch.no_grad():
                for name, param in model.named_parameters():
                    uu[name] = param.grad.data.detach().clone()
                    ws[name] = param.data.detach().clone()

                for i in peers:
                    COMM.bsend(ws, dest=i)
             
                for name in ws.keys():
                    ws[name] = torch.zeros_like(ws[name])
                for i in peers:
                    ws_received_ = COMM.recv(source = i)
                    for name in ws.keys():
                        ws[name] += ws_received_[name].to(device)/len(peers)
                
                lr = optimizer.get_lr()
                for name, param in model.named_parameters():
                    ws[name] -= lr*yy[name] # step
                    ws[name] = torch.max(abs(ws[name])-lr*l1,torch.zeros_like(ws[name]))*torch.sign(ws[name]) # proximal norm1
                    param.copy_(ws[name])
                    
                    yy[name] -= dd[name]
                    
            model.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward() # calculate v
            if CLIP_GRAD:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM_MAX)
                 
            pred = output.argmax(dim=1, keepdim=True)
            acc  = 100.*pred.eq(target.view_as(pred)).sum().item()/ len(target);acc=torch.tensor(acc)
            train_stats.append({'loss':float(loss),'correct':float(acc)})
             
            for name, param in model.named_parameters():
                dd[name] = (1-beta)*(dd[name] - uu[name]) + param.grad.data.detach().clone()
                yy[name] += dd[name]
             
            for i in peers:
                COMM.bsend(yy, dest=i)
         
            for name in ws.keys():
                yy[name] = torch.zeros_like(ws[name])
            for i in peers:
                yy_received_ = COMM.recv(source = i)
                for name in ws.keys():
                    yy[name] += yy_received_[name].to(device)/len(peers)
               
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
 
