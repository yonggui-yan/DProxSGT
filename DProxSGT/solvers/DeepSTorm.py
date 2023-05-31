from .torch_backend import *
from .core import *

##################################### DeepSTorm ##################################################
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
