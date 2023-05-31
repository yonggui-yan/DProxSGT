from .torch_backend import *
from .core_compress import *
from .compressor import *
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
        
        
