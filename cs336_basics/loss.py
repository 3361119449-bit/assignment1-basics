# losses.py             # cross_entropy
import torch

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




