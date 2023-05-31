from .torch_backend import *
from .core import *
##################################### AllReduce ##################################################
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
