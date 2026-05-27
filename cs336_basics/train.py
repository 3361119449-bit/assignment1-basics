import torch
import math
from einops import rearrange, einsum
import numpy as np


from cs336_basics.m import TransformerLM,softmax




import os
import csv
import json
import time
import math
# 4 Training a Transformer LM 训练 Transformer LM
# 现在我们已经有了预处理数据（通过 tokenizer）和模型（Transformer）的步骤。剩下的是
# 构建所有用于支持训练的代码。这包括：
# • Loss：我们需要定义损失函数（cross-entropy）。
# • Optimizer：我们需要定义优化器来最小化该损失（AdamW）。
# • Training loop：我们需要所有支持基础设施，用于加载数据、保存 checkpoints 和管理训
# 练。


# logits: (..., vocab_size)
# targets: (...)
def cross_entropy(logits,target_x):
    # 它接收 predicted logits（oi）和 targets（xi+1）
#     你的函数应处
# 理以下内容：
# • 为了数值稳定性，减去最大元素。
# • 尽可能抵消 log 和 exp。
# • 处理任何额外 batch dimensions，并返回 batch 上的平均值。和 Section 3.2 一样，
# 我们假设 batch-like dimensions 总是位于词表大小维度之前。


# log_softmax = logits - logsumexp(logits)
# logits[target_x]:取出真实token对于的输出分数
    m,_=torch.max(logits,dim=-1,keepdim=True)
    #这里的 max返回 不是 tensor，而是一个 tuple。
    exp_sum=torch.sum(torch.exp((logits-m)),dim=-1,keepdim=True)
    log_p=(torch.gather(logits, dim=-1, index=target_x.unsqueeze(-1))-m)-torch.log(exp_sum)
    l=-torch.mean(log_p.squeeze(-1))#所有维度平均
    return l
#    l_i=-torch.mean(log_p.squeeze(-1),dim=-1)#所有维度平均
    
#     l=torch.mean(l_i)


# max 和 sum 要用 torch.max / torch.sum
# logits[target_x.unsqueeze(-1)]
# PyTorch 会理解成：

# text


# 用 target_x 里的值去索引 logits 的第 0 维






# softmax(o_i)[x_{i+1}]
# 也就是：模型给真实下一个 token 分配了多少概率。
# 所以目标 x 的作用是：

# 作为索引，从模型预测的 vocab 分布里选出真实 token 的概率
# cross entropy 本质上是在惩罚：

# 模型没有把高概率分给真实下一个 token


# 4.2.1 Implementing SGD in PyTorch 在 PyTorch 中实现 SGD

# 应初始化你的 optimizer。这里，params 将是要优化的
# 参数集合（或 parameter groups，如果用户希望对模型不同部分使用不同 hyperparameters，例如不同 learning rates）。确保将 params 传给 base class 的 __init__ 方法，它
# 会存储这些参数以供 step 使用。你可以根据 optimizer 接收额外参数（例如 learning
# rate 是常见的参数），并将它们作为一个 dictionary 传给 base class constructor，其中
# keys 是你为这些参数选择的名称（strings）



# 为什么叫 parameter groups？

# 因为用户可以传不同参数组，用不同学习率：

# [
#   {"params": embedding_params, "lr": 1e-4},
#   {"params": transformer_params, "lr": 1e-3},
# ]
# PyTorch 会帮你管理参数组，之后 step 里从 self.param_groups 取出来更新。

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
# 这个 SGD 示例不是标准 SGD学习率会随每个参数的更新次数衰减
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

# closure: Optional[Callable] = None这是个啥
# closure 是一个可选的函数，优化器可以调用它来重新计算 loss。（LBFGS）

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

# 4.3 AdamW
# AdamW 使用额外内存来换取更好的稳定性和收敛
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
# torch.no_grad() / .data
# 你当前 .data 能表达逻辑，但更稳的 PyTorch 风格是把参数更新放进 torch.no_grad()。
# 不过如果作业不严格检查这个，主要影响风格，不是公式。



# 4.4 Learning rate scheduling 学习率调度
def learning_rate_schedule(t,alpha_max,alpha_min,T_w,T_c):
#     t: 当前 step / iteration
# alpha_max: 最大学习率
# alpha_min: 最小学习率
# T_w: warmup steps
# T_c: cosine decay 总步数
    if T_c<=T_w:
        raise ValueError("T_c<=T_w")

    
    if t<T_w:
        return t*alpha_max/T_w
    elif t>T_c:
        return alpha_min
    else:
        return alpha_min+(1+math.cos((t-T_w)*math.pi/(T_c-T_w)))*(alpha_max-alpha_min)/2

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
# 第一遍算完 total 后，第二个 for p in parameters 可能已经没有东西了。

# 5 Training loop 训练循环
# 现在，我们终于要把目前构建的主要组件组合起来：tokenized data、model 和 optimizer

# 5.1 Data Loader 数据加载器


# data loader 从长 token 序列里随机切长度为 m 的片段作为输入，再取紧跟着右移一位的片段作为答案。




# Deliverable： 编写一个函数，接收一个 numpy array x（包含 token ID 的 integer
# array）、一个 batch_size、一个 context_length 和一个 PyTorch device string（例
# 如 'cpu' 或 'cuda:0'），并返回一对 tensors：采样得到的 input sequences 和对应的
# next-token targets。两个 tensors 的形状都应为 (batch_size, context_length)，包
# 含 token IDs，并且都应放置在请求的 device 上。
# x = np.array([
#   101, 55, 92, 50256,
#   77, 88, 12, 9, 50256,
#   300, 301, 302
# ])
# 返回eg:
# inputs =
# [
#   [x2,  x3,  x4 ],
#   [x10, x11, x12]
# ]

# targets =
# [
#   [x3,  x4,  x5 ],
#   [x11, x12, x13]
# ]

# dataset.shape = (num_tokens,)
# 那么：

# text


# dataset[indices]
# 表示：

# text


# 从一维数组里取 indices 指定的位置
# 结果 shape 就是 indices.shape：

# text


# (batch_size, context_length)
def data_loading(x,batch_size,context_length,device=None):
#     x: 一整条 token 序列，shape 通常是 (num_tokens,)
# batch_size: B，一次取多少条序列
# context_length: m，每条输入序列长度
# device: 返回 tensor 放到哪个设备
    starts = np.random.randint(0,len(x)-context_length,batch_size).reshape(-1,1)
    offsets=np.arange(context_length)
    indices = starts + offsets
    

    inputs = torch.from_numpy(x[indices]).to(device=device, dtype=torch.long)
    labels = torch.from_numpy(x[indices+1]).to(device=device, dtype=torch.long)

    return inputs,labels

# 随机选 batch_size 个起点
# inputs 取 dataset[start : start + context_length]
# labels 取 dataset[start + 1 : start + context_length + 1]





# 这段意思是：tokenized 数据可能非常大，不能一次全部读进 RAM，所以用 memory map 假装它在内存里，但实际是按需从磁盘读取。
# x = np.load("tokens.npy", mmap_mode="r")
# 你可以照常写：

# x[1000:2000]
# 但它只会从磁盘加载这一小段。
# 一句话总结：
# 训练时不要把整个 token 数据读进内存；
# 用 mmap 按需读取；
# data loader 每次只采样小 batch；
# 并检查读出来的 token id 是否合理。0 <= token_id < 10000




# 5.2 Checkpointing Checkpoint 保存与加载
# 除了加载数据，我们还需要在训练时保存模型。运行 jobs 时，我们常常希望能够恢复一个中
# 途停止的 training run（例如由于 job 超时、机器故障等）。



# 应将 model、optimizer 和 iteration 的所有 state dump 到 file-like object out 中。
# 你可以使用 model 和 optimizer 的 state_dict 方法获得它们相关 state，并使用
# torch.save(obj, out) 将 obj dump 到 out 中（PyTorch 这里支持 path 或 file-like
# object）。典型选择是让 obj 成为一个 dictionary，但只要之后能加载 checkpoint，你可
# 以使用任意格式。

# 保存时一般会保存三样东西：

# model 的 state_dict
# optimizer 的 state_dict
# iteration
def save_checkpoint(model, optimizer, iteration, out):
#     model: torch.nn.Module
# optimizer: torch.optim.Optimizer
# iteration: int
# out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model_dict=model.state_dict()
    # 每个 nn.Module 都有一个 state_dict() 方法，它返回一个包含所有 learnable weights 的 dictionary；
# 稍后可以用对应的 load_state_dict()方法恢复这些 weights。
    optimizer_dict=optimizer.state_dict()
    obj={"model":model_dict,"optimizer":optimizer_dict,"iteration":iteration}
    torch.save(obj,out)
# 可以将一个 object（例如一个包含 tensors 和普通 Python objects 如 integers 的
#  dictionary）dump 到文件（path）或 file-like object，之后可以用 torch.load(src) 加载回内存。




# 所以这段话的人话版是：

# save_checkpoint(model, optimizer, iteration, out)
# 需要把 model 参数、optimizer 状态、iteration 打包保存到 out。
# 保存格式你自己定，但 load_checkpoint 必须能按同样格式读回来。






def load_checkpoint(src, model, optimizer):
#     src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
# model: torch.nn.Module
# optimizer: torch.optim.Optimizer
    checkpoint=torch.load(src)
    model_dict=checkpoint["model"]
    optimizer_dict=checkpoint["optimizer"]
    iteration=checkpoint["iteration"]


    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)
    return iteration




# 概念流程：

# checkpoint = 从 src 加载出来的字典

# model.load_state_dict(checkpoint 里的 model state)

# optimizer.load_state_dict(checkpoint 里的 optimizer state)

# return checkpoint 里的 iteration




# 代码可以简化

# 你的代码可以直接写成：

def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

# 即使 checkpoint 原来在 GPU 保存，也能先安全加载到 CPU。
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]





class ExperimentLogger:
    def __init__(self, log_dir: str, config: dict, append: bool = False):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_path = os.path.join(log_dir, "metrics.csv")
        self.config_path = os.path.join(log_dir, "config.json")
        self.start_time = time.perf_counter()

        self.fieldnames = [
            "step",
            "wallclock_time",
            "tokens_processed",
            "lr",
            "train_loss",
            "val_loss",
            "val_perplexity",
        ]

        # config 每次覆盖一般没问题，因为同一个实验配置应该一致
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        # 如果 append=True 且 metrics.csv 已存在，就不重写 header
        if append and os.path.exists(self.metrics_path):
            pass
        else:
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _write_row(self, row: dict):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def log_train(self, step, loss, lr, batch_size, context_length):
        wallclock_time = time.perf_counter() - self.start_time
        tokens_processed = step * batch_size * context_length

        row = {
            "step": step,
            "wallclock_time": wallclock_time,
            "tokens_processed": tokens_processed,
            "lr": lr,
            "train_loss": loss,
            "val_loss": "",
            "val_perplexity": "",
        }

        self._write_row(row)

        print(
            f"step {step}: train_loss={loss:.4f}, "
            f"lr={lr:.6e}, time={wallclock_time:.1f}s",
            flush=True,
        )

    def log_val(self, step, val_loss, lr, batch_size, context_length):
        wallclock_time = time.perf_counter() - self.start_time
        tokens_processed = step * batch_size * context_length
        val_perplexity = math.exp(val_loss)

        row = {
            "step": step,
            "wallclock_time": wallclock_time,
            "tokens_processed": tokens_processed,
            "lr": lr,
            "train_loss": "",
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
        }

        self._write_row(row)

        print(
            f"step {step}: val_loss={val_loss:.4f}, "
            f"ppl={val_perplexity:.4f}, time={wallclock_time:.1f}s",
            flush=True,
        )





# 5.3 Training loop 训练循环

# 现在，终于到了把你已经实现的所有组件组合到主训练脚本中的时候。让你的训练脚本容易
# 以不同 hyperparameters 启动 training runs（例如通过 command-line arguments 接收它
# 们）会带来回报，因为之后你会多次进行这些实验来研究不同选择如何影响训练。




# Deliverable： 编写一个脚本，它运行 training loop，在用户提供的输入上训练你的模
# 型。具体而言，我们建议你的训练脚本至少允许以下内容：
# • 能够配置和控制各种 model 与 optimizer hyperparameters。
# • 使用 np.memmap 对大型训练和验证数据集进行内存高效加载。
# • 将 checkpoints 序列化到用户提供的路径。
# • 周期性记录 training 和 validation performance（例如记录到 console 和/或
# Weights and Biasesa 这样的外部服务）。
def training_together(
    train_data_path,
    val_data_path,
    checkpoint_path,

    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,

    batch_size,
    total_steps,

    alpha_max,
    alpha_min,
    warmup_steps,
    cosine_steps,

    betas,
    eps,
    weight_decay,
    max_grad_norm,

    log_interval,
    eval_interval,
    checkpoint_interval,
    val_batches,

    device,
    dtype=None,
    resume_from=None,
):
    #日志
    log_dir = os.path.dirname(checkpoint_path)
    if log_dir == "":
        log_dir = "."

    config = {
        "train_data_path": train_data_path,
        "val_data_path": val_data_path,
        "checkpoint_path": checkpoint_path,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "warmup_steps": warmup_steps,
        "cosine_steps": cosine_steps,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "device": str(device),
        "dtype": str(dtype),
    }

    logger = ExperimentLogger(
    log_dir,
    config,
    append=resume_from is not None,
)


    train_data=np.load(train_data_path, mmap_mode="r")
    val_data=np.load(val_data_path, mmap_mode="r")
# 这样不会把整个 tokenized dataset 加载进内存。
   



    model=TransformerLM(d_model,num_heads,d_ff,rope_theta,
                 vocab_size,context_length, num_layers,
                 dtype=dtype,device=device)
    
    optimizer=AdamW(model.parameters(),lr=alpha_min,betas=betas,eps=eps,weight_decay=weight_decay)
# 其实 optimizer 初始 lr 不太重要，因为你每一步都会覆盖
 

    # load_checkpoint(checkpoint_path, model, optimizer)
 
    t=1
    if resume_from:
        t=load_checkpoint(resume_from, model, optimizer)+1
    model.train()
    while t <=total_steps:
        lr=learning_rate_schedule(t,alpha_max,alpha_min,warmup_steps,cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs,labels=data_loading(train_data,batch_size,context_length,device=device)



        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs),labels)
        loss.backward() 
        gradient_clipping(model.parameters(),max_grad_norm)#max_grad_norm 如果可能为 None，需要判断
        optimizer.step()
        

        if t % checkpoint_interval == 0:#定期存档
            save_checkpoint(model, optimizer, t, checkpoint_path)

        if t % log_interval == 0:#定期打印训练损失
            # print(t,val_loss,math.exp(val_loss))
            # print(f"step {t}: val_loss={loss:.4f}, ppl={math.exp(loss):.4f}")
            #日志
            logger.log_train(
                step=t,
                loss=loss.item(),
                lr=lr,
                batch_size=batch_size,
                context_length=context_length,
            )

        if t % eval_interval==0:#定期验证集验证
            model.eval()
            val_loss=0
            with torch.no_grad():
                for _ in range(val_batches):
                    val_inputs,val_labels=data_loading(val_data,batch_size,context_length,device=device)
                    val_loss+=(cross_entropy(model(val_inputs),val_labels)).item()
            val_loss = val_loss/val_batches
            
            # print(t,val_loss,math.exp(val_loss))
            # print(f"step {t}: train_loss={loss.item():.4f}, lr={lr:.6e}")
             #日志
            logger.log_val(
                step=t,
                val_loss=val_loss,
                lr=lr,
                batch_size=batch_size,
                context_length=context_length,
            )

            model.train()
        
        t+=1
    save_checkpoint(model, optimizer, t-1, checkpoint_path)#保存已经完成的最后一步：
    return model,optimizer



#                 跑 val_batches 个 validation batch：
# val_loss = 多个 validation batch loss 的平均
# val_ppl = exp(val_loss)
# model.train()
# 打印：



# resume_from 表示：从哪个 checkpoint 继续训练。
# checkpoint_path 是你训练过程中保存 checkpoint 的位置。





# 注意 logits shape 是：

# (batch, context_length, vocab_size)
# targets shape 是：

# (batch, context_length)













# 6 Generating text 生成文本

# prompt是字符串
# dtype = torch.long
def Decoding(model,tokenizer,prompt,max_generated_tokens,temperature=1,Top_p=None,device="cpu"):
#     • 为用户提供的 prompt 生成 completions（即接收某些 x1...t 并采样 completion，直
# 到遇到 <|endoftext|> token）。
# • 允许用户控制最大 generated tokens 数量。
# • 给定 desired temperature value，在采样前对 predicted next-token distributions
# 应用 softmax temperature scaling。
# • Top-p sampling（[A. Holtzman et al., 2020] 也称 nucleus sampling），给定用户指
# 定的 threshold value。
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prompt)
        input=torch.tensor(ids,dtype = torch.long,device=device).reshape(1,-1)
        # x = torch.cat([x, new_row], dim=0)

        num_tokens=0
        while num_tokens <max_generated_tokens:
            logits=model(input)
            out=softmax(logits/temperature,-1)
            p_tokens=out[:,-1,:][0]#取出最后一个词的概率分布
            if Top_p is not None:
                values,indices=torch.sort(p_tokens,descending=True)
                p_cumsum=torch.cumsum(values, dim=0)
                i=0
                while  i < p_cumsum.numel() and p_cumsum[i]<=Top_p :
                    i+=1
                kept_probs=values[:i+1]
                kept_indices = indices[:i+1]
                sampled_pos = torch.multinomial(kept_probs / kept_probs.sum(), num_samples=1)
                next_id = kept_indices[sampled_pos].reshape(1,1)
                
                


            else:
                next_id = torch.multinomial(p_tokens, num_samples=1).reshape(1,1)
                # _,next_id=torch.max(p_tokens,dim=0)
            input=torch.cat([input, next_id], dim=1)

            num_tokens+=1
            if tokenizer.vocab[next_id.item()]==b"<|endoftext|>":    #注意先编码
                break
    
    # return tokenizer.decode(list(input[0]))
    return tokenizer.decode(input[0].tolist())
# input 是二维 tensor，decode 通常要一维 list。

        #         byte_sequence = b"".join(vocab[token_id] for token_id in ids)

        # text = byte_sequence.decode("utf-8", errors="replace")#先收集 bytes pieces，再一次性 join。