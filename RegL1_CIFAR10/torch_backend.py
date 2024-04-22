import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = MPI.COMM_WORLD.Get_size()
UID  = MPI.COMM_WORLD.Get_rank()

import torch
import torchvision
from torch import nn

import time
from collections import namedtuple
from functools import singledispatch
 
class Timer:
    def __init__(self, synch=None,total_time=0.0):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.time()]
        self.total_time = total_time

    def __call__(self, include_in_total=True):
        self.synch() # sync
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t
        
localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

class TableLogger: ## used for print on  the  screen
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            if UID == 0: print(*(f'{k:>12s}' for k in self.keys),flush=True)
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered),flush=True)

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class StatsLogger():
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            #v.append(output[k].detach())
            v.append(output[k])

    def stats(self, key):
        return self._stats[key]
        
    def mean(self, key):
        return np.mean(np.array(self.stats(key)), dtype=np.float)
###############################################################

 
###############################################################
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if UID <= (SIZE//torch.cuda.device_count()):
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")

trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters()) 

class TorchOptimiser:
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())

    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k, v in self.opt_params.items()}

    def get_lr(self):
        return self._opt.param_groups[0]['lr']

    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self):
        return repr(self._opt)


def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False,step_number=0):
    return TorchOptimiser(weights, torch.optim.SGD, step_number=step_number, lr=lr, momentum=momentum,
                          weight_decay=weight_decay, dampening=dampening,
                          nesterov=nesterov)
           
