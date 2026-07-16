
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:#这个参数当前有没有算出来的梯度还没调用 loss.backward()
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults={"lr": lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            betas=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                # m=state.get("m",torch.zeros_like(p.data))
                # v=state.get("v",torch.zeros_like(p.data))
                # t=state.get("t",0)+1
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    # state["m"] = torch.zeros_like(p,dtype=torch.float32)
                    # state["v"] = torch.zeros_like(pdtype=torch.float32)

                t=state["t"]+1
                m=state["m"]
                v=state["v"]
                
                grad = p.grad
                #grad = p.grad.data
                lr_t=lr*math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                with torch.no_grad():
                    p-=lr*weight_decay*p
                #p.data-=lr*weight_decay*p.data
                # m=betas[0]*m+(1-betas[0])*grad
                # v=betas[1]*v+(1-betas[1])*(grad**2)
                # state["m"]=m
                # state["v"]=v
                    m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                    p-=lr_t*m/(torch.sqrt(v)+eps)
                    # p.data-=lr_t*m/(torch.sqrt(v)+eps)
                
                state["t"]=t

        return loss
    
# 4.5 Gradient clipping 梯度裁剪
# 你现在：
#     每个 p.grad 单独看 norm

# 作业要：
#     把所有 p.grad 看成一个大向量，算总 norm
def gradient_clipping(parameters,max_l2_norm):
    total=0
    p_list=[]
    for p in parameters:
        if p.grad is None:
            continue
        p_list.append(p)
        total+=torch.sum(p.grad**2)#数值太大？
    if not p_list:
        return None
    l2=torch.sqrt(total)
    if l2>=max_l2_norm:
        for p in p_list:
            if p.grad is None:
                continue
            with torch.no_grad():
                p.grad*=max_l2_norm/(l2+1e-6)
