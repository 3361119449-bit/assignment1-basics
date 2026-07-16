import os

import numpy as np
import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.loss import cross_entropy
from cs336_basics.optim import AdamW, gradient_clipping
from cs336_basics.schedules import learning_rate_schedule
from cs336_basics.data import data_loading
from cs336_basics.checkpointing import save_checkpoint as save_checkpoint_fn, load_checkpoint
from cs336_basics.logging_utils import ExperimentLogger


# 5.3 Training loop 训练循环


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
    save_intermediate_checkpoints=True,
    save_final_checkpoint=True,
    use_rmsnorm=True,
    norm_position="pre",
    use_rope=True,
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
        "save_intermediate_checkpoints": save_intermediate_checkpoints,
        "save_final_checkpoint": save_final_checkpoint,
        "use_rmsnorm": use_rmsnorm,
        "norm_position": norm_position,
        "use_rope": use_rope,
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
                 dtype=dtype,device=device,use_rmsnorm=use_rmsnorm,
                 norm_position=norm_position,use_rope=use_rope)
    model = torch.compile(model)
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
        
        if save_intermediate_checkpoints:
            if t % checkpoint_interval == 0:#定期存档
                save_checkpoint_fn(model, optimizer, t, checkpoint_path)

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
    if save_final_checkpoint:
        save_checkpoint_fn(model, optimizer, t-1, checkpoint_path)#保存已经完成的最后一步：
    return model,optimizer
