"""Warm nonpreemptive kernels without changing the starting learning state."""
from copy import deepcopy
import time
import torch
from src.schedulers.boundary_scheduler import evaluation_state


def warm_compute(model,optimizer,device,rounds=5):
    saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    saved_optimizer=deepcopy(optimizer.state_dict())
    durations=[]
    with evaluation_state(model):
        model.train()
        dummy=torch.zeros(128,3,32,32,device=device)
        labels=torch.zeros(128,dtype=torch.long,device=device)
        for _ in range(rounds):
            if device.type=='cuda':torch.cuda.synchronize(device)
            started=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(dummy),labels)
            loss.backward();optimizer.step()
            if device.type=='cuda':torch.cuda.synchronize(device)
            durations.append(time.perf_counter()-started)
        model.load_state_dict(saved);optimizer.load_state_dict(saved_optimizer)
        optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
    return durations
