

import torch
import math
from einops import rearrange, einsum
# 实现一个继承自 torch.nn.Module 的 Linear 类，并执行线性变换。你的
# 实现应遵循 PyTorch 内置 nn.Linear 模块的接口，除了没有 bias argument 或 parameter。



# 确保：
# • subclass nn.Module
# • 调用 superclass constructor
# • 构造并存储你的参数为 W（不是 W⊤），并将它放入 nn.Parameter
# • 当然，不要使用 nn.Linear 或 nn.functional.linear
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
    #     in_features: int 输入的最后一维。
    # out_features: int 输出的最后一维。
    # device: torch.device | None = None 存储参数的设备。
    # dtype: torch.dtype | None = None 参数的数据类型。
        self.weight = torch.nn.Parameter(
                    torch.empty(
                        out_features,
                        in_features,
                        device=device,
                        dtype=dtype,
                    )
                )
        
        # 作业要求：
        # Linear weights: N(0, 2 / (din + dout))，并截断到 [-3σ, 3σ]
        std = math.sqrt(2 / (in_features + out_features))

        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        # self.weight=torch.zeros(out_features,in_features,dtype).to(device)
        # torch.nn.init.trunc_normal_(self.weight, std=0.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
            




# 如上所述，Transformer 的第一层是 embedding layer，它将整数 token ID 映射到维度为
# d_model 的向量空间。。forward 方法应该使用形状为 (batch_size, sequence_length)
# 的 token ID torch.LongTensor 对形状为 (vocab_size, d_model) 的 embedding matrix 建
# 索引，从而为每个 token ID 选择 embedding vector。


class Embedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
#         num_embeddings: int 词表大小。
# embedding_dim: int embedding vectors 的维度，即 dmodel。
# device: torch.device | None = None 存储参数的设备。
# dtype: torch.dtype | None = None 参数的数据类型。
        self.weight=torch.nn.Parameter(torch.empty(num_embeddings,embedding_dim,
                                                      device=device,dtype=dtype))
        #这里应该是 Parameter 类，不是小写 parameter 函数。
        #empty 是 torch.empty，不是 torch.nn.empty。
        #初始化 weight
        torch.nn.init.trunc_normal_( self.weight,
                                        mean=0.0,
                                        std=1.0,
                                        a=-3.0 ,
                                        b=3.0 )

# Embedding 的作用是：把离散的 token id 变成连续向量，模型后面的层才能处理。
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
#         查找给定 token ID 的 embedding vectors。
        assert token_ids.dtype == torch.long

        num_embeddings = self.weight.shape[0]
        embedding_dim = self.weight.shape[1]

        assert token_ids.max().item() < num_embeddings
        assert token_ids.min().item() >= 0

        output = self.weight[token_ids]

        assert output.shape[-1] == embedding_dim

        return output
    
# 你要检查三点：

# token_ids 应该是整数类型，通常是 torch.long。
# token_ids 里的最大值必须小于 vocab_size。
# 输出最后一维应该等于 embedding_dim。


    # 确保：
# 20
# • subclass nn.Module
# • 调用 superclass constructor
# • 将 embedding matrix 初始化为 nn.Parameter
# • 存储 embedding matrix 时使 d_model 位于最后一维
# • 当然，不要使用 nn.Embedding 或 nn.functional.embedding
# 同样，使用上面的初始化设置，并用 torch.nn.init.trunc_normal_ 初始化权重。





# in_dtype = x.dtype
# x = x.to(torch.float32)
# # Your code here performing RMSNorm
# ...
# result = ...
# # Return the result in the original dtype
# return result.to(in_dtype
                 

class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
#         d_model: int 模型的隐藏维度。
# eps: float = 1e-5 用于数值稳定性的 epsilon 值。
# device: torch.device | None = None 存储参数的设备。
# dtype: torch.dtype | None = None 参数的数据类型。
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.device=device
        self.dtype=dtype
        self.weight=torch.nn.Parameter(torch.ones(self.d_model,device=device, dtype=dtype))




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #处理形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回相同形状的张量。
# Note: 记得在执行 normalization 之前将输入 upcast 到 torch.float32
# （之后再 downcast 回原始 dtype），如上所述。
        in_dtype=x.dtype
        x=x.to(torch.float32)
        RMS_a=torch.sqrt(einsum(x,x,"... d_model, ... d_model -> ...")/self.d_model
                        +self.eps).unsqueeze(-1)
        #应该先得到 ...，然后用 .unsqueeze(-1) 补出最后一维。
        
        #math.sqrt(...) 不能用于 tensor，要用 torch.sqrt(...)




        result=einsum((x/RMS_a),self.weight.to(torch.float32),
                      "... d_model, d_model -> ... d_model")
        # 等价于result = x / RMS_a* self.weight


        return result.to(in_dtype)



# #在这个特定情况下，你可以在实现中使用 torch.sigmoid 以获得数值稳定性。
# 你应该在实现中将 df f 设置为约 3
# 8 × dmodel，同时确保内层 feed-forward layer 的维度
# 是 64 的倍数，以便充分利用硬件。


class Positionwise_feedforward(torch.nn.Module):
    def __init__(self,d_model,d_ff=None,device=None,dtype=None):
        super().__init__()
        if d_ff is None:
            self.d_ff = int(((8 / 3 * d_model) + 32) // 64 * 64)
        else:
            self.d_ff=d_ff
        self.weight_1=torch.nn.Parameter(torch.empty(self.d_ff,d_model,
                                                     device=device,dtype=dtype))
        self.weight_2=torch.nn.Parameter(torch.empty(d_model,self.d_ff,
                                                     device=device,dtype=dtype))
        self.weight_3=torch.nn.Parameter(torch.empty(self.d_ff,d_model,
                                                     device=device,dtype=dtype))
        
        std = math.sqrt(2 / (self.d_ff + d_model))

        torch.nn.init.trunc_normal_(
            self.weight_1,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            self.weight_2,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            self.weight_3,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #处理形状为 (batch_size, sequence_length, d_model) 的输入张量
        w1x=einsum(x,self.weight_1,
                             "... d_model,d_ff d_model -> ... d_ff")
        x=w1x*torch.sigmoid(w1x)*einsum(x,self.weight_3,
                            "... d_model,d_ff d_model -> ... d_ff")  
        return einsum(x,self.weight_2,
                        "... d_ff, d_model d_ff-> ... d_model")
    

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 构造 RoPE 模块并在需要时创建 buffer。
        # theta: float RoPE 的 Θ 值。
        # d_k: int query 和 key vectors 的维度。
        # max_seq_len: int 输入的最大序列长度。
        # device: torch.device | None = None 存储 buffer 的设备。

        #错误一：双for赋值
        # table=torch.empty(max_seq_len,d_k//2,2,device=device)
        # for i in range(max_seq_len):
        #     for k in range(d_k//2):
        #         table[i,k,0]=math.sin(i/(theta**((2*k)/d_k)))
        #         table[i,k,1]=math.cos(i/(theta**((2*k)/d_k)))

        #错误二，不会创建tensor
        # positions=[i for i in range(max_seq_len)].reshape(-1,1)
        # inv_freq=[1/(theta**((2*k)/d_k)) for k in range(d_k//2)].reshape(1,-1)

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")
        positions=torch.arange(max_seq_len,device=device).float().reshape(-1,1)
        inv_freq=1/(theta**(torch.arange(0,d_k,2,device=device).float()/d_k))
        inv_freq=inv_freq.reshape(1,-1)
        # x = torch.arange(0, d_k, 2)它默认是整数类型：


        table=inv_freq*positions
        sin_table=torch.sin(table)
        cos_table=torch.cos(table)

        
        self.register_buffer("sin_table", sin_table, persistent=False)
        self.register_buffer("cos_table", cos_table, persistent=False)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#     处理形状为 (..., seq_len, d_k) 的输入张量，并返回相同形状的张量。注意你应该容忍
# x 具有任意数量的 batch 维度。你应该假设 token positions 是形状为 (..., seq_len)
# 的张量，指定 x 沿 sequence 维度的 token positions。
# 你应该使用 token positions 来沿 sequence 维度切片你的（可能预计算的）cos 和 sin
# 张量。

#在 forward 里使用 buffer 前，把 sin/cos 转成和 x 一样的 dtype。以及device

# 标准实现里有些会一直用 float32 计算旋转，最后再 cast 回输入 dtype，以提高精度。


        # seq_len=x.shape[-2]
        # sin_table = self.sin_table[:seq_len].to(dtype=x.dtype,device=x.device)
        # cos_table = self.cos_table[:seq_len].to(dtype=x.dtype,device=x.device)
#你应该假设 token positions 是形状为 (..., seq_len)#是指输入的序列在原来文本中的真实位置，而不是每次从0开始
        # 3. 检查 buffer device
        if self.sin_table.device != x.device or self.cos_table.device != x.device:
            raise RuntimeError(
                f"RoPE buffers are on {self.sin_table.device}, but x is on {x.device}. "
                f"Call rope.to(x.device), or construct RoPE with device=x.device."
            )

        # 4. token_positions 必须放到同一个 device，并且用于索引时应该是 long
        token_positions = token_positions.to(device=x.device, dtype=torch.long)


  
        sin_table = self.sin_table[token_positions].to(dtype=x.dtype,device=x.device)#注意切片方式
        cos_table = self.cos_table[token_positions].to(dtype=x.dtype,device=x.device)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        while sin_table.ndim < x_even.ndim:#多头情况或者多batch——like
            sin_table = sin_table.unsqueeze(-3)
            cos_table = cos_table.unsqueeze(-3)



        x_even_new = x_even * cos_table - x_odd* sin_table
        x_odd_new = x_even * sin_table + x_odd * cos_table
        #采用作业里的旋转


        result=torch.empty_like(x)
        result[..., 0::2] = x_even_new
        result[..., 1::2] = x_odd_new
        return result



# register_buffer 是 torch.nn.Module 提供的方法。
# self.register_buffer(name, tensor, persistent=False)
# 等价于注册一个模块状态。

# 几个参数的意思是：

# name：这个 buffer 的名字，字符串，比如 "cos_cached"。

# tensor：你要保存的 tensor。

# persistent=False：不要把它保存进 state_dict()。
# 注册之后，你可以像普通属性一样访问：

# self.cos_cached

# 当作模块状态管理。是什么意思

# 模块状态管理”就是让 PyTorch 帮你追踪这个 tensor，让它跟着模型设备/dtype 变化，并按需要参与或不参与保存。



def softmax(x,i):

# Deliverable： 编写一个函数，对张量应用 softmax 操作。你的函数应接受两个参数：
# 一个张量和一个维度 i，并对输入张量的第 i 个维度应用 softmax。输出张量应与输入形
# 状相同，但其第 i 个维度现在是一个归一化概率分布。使用从第 i 个维度所有元素中减去
# 该维度最大值的技巧，以避免数值稳定性问题。
    values, _ =torch.max(x,dim=i,keepdim=True)
    exp_x=torch.exp(x-values)
    sum_x=exp_x.sum(dim=i,keepdim=True)
    return exp_x/sum_x



#作业里序列长度一样，是因为实现的是同一段序列内部的 self-attention；
# 如果是 cross-attention，Q 的序列长度就可以和 K/V 不一样。
#你可以把 d_k 看成每个点的坐标维度；有 n 个查询点、m 个被查询点。
#每个查询点都和所有被查询点沿坐标维度做点积，得到 n × m 个相似度分数。
# 点积注意力不是直接在找距离最近的点，而是在找向量表示最相似的点。
def scaled_dot_product_attention(queries,key,values,boolean_mask=None):
# Deliverable： 实现 scaled dot-product attention 函数。你的实现应处理形状为
# (batch_size, ..., seq_len, d_k) 的 keys 和 queries，以及形状为 (batch_size,
# ..., seq_len, d_v) 的 values，其中 ... 表示任意数量的其他 batch-like 维度（如
# 果提供）。实现应返回形状为 (batch_size, ..., seq_len, d_v) 的输出。关于 batchlike 维度的讨论见 Section 3.2。
# 你的实现还应支持用户提供的可选 boolean mask，形状为 (seq_len, seq_len)。mask
# 值为 True 的位置的 attention probabilities 应总体求和为 1，而 mask 值为 False 的位
# 置的 attention probabilities 应为 0。
    d_k=key.shape[-1]
    scores=einsum(queries,key,"... seq_len_q d_k,... seq_len_k d_k -> ... seq_len_q seq_len_k")/math.sqrt(d_k)
    if boolean_mask is not None:
        # scores = scores.masked_fill(boolean_mask == False, float("-inf"))
        scores = scores.masked_fill(~boolean_mask, float("-inf"))
    attention=softmax(scores,-1)
    return einsum(attention,values,"... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    





# 如果没有 W_O，每个 head 的输出就只是固定拼接，head 之间没有可学习的线性混合。
class MultiHeadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(self,d_model: int ,num_heads: int,theta: float,max_seq_len: int,
                 dtype=None,device=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.weight_q=torch.nn.Parameter(torch.empty(self.d_model,
                                        self.d_model,dtype=dtype,device=device))
        # torch.nn.init.trunc_normal_(self.weight_q)
        self.weight_k=torch.nn.Parameter(torch.empty(self.d_model,
                                        self.d_model,dtype=dtype,device=device))
        self.weight_v=torch.nn.Parameter(torch.empty(self.d_model,
                                        self.d_model,dtype=dtype,device=device))
        self.weight_o=torch.nn.Parameter(torch.empty(self.d_model,
                                        self.d_model,dtype=dtype,device=device))
        
        std = math.sqrt(1 / (self.d_model))

        torch.nn.init.trunc_normal_(
            self.weight_q,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            self.weight_k,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            self.weight_v,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            self.weight_o,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.rope=RoPE(theta, self.d_k, max_seq_len, device)

#causal self-attention 通常允许看自己：j <= i
    def forward(self,x,token_positions=None):
        if token_positions is None:
            token_positions=torch.arange(x.shape[-2],device=x.device)
        q=einsum(self.weight_q,x,"d_model_q d_model , ... d_model ->... d_model_q")
        k=einsum(self.weight_k,x,"d_model_k d_model , ... d_model ->... d_model_k")
        v=einsum(self.weight_v,x,"d_model_v d_model , ... d_model ->... d_model_v")

        q=rearrange(q,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads=self.num_heads,d_k=self.d_k)
        k=rearrange(k,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads=self.num_heads,d_k=self.d_k)
        v=rearrange(v,"... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",num_heads=self.num_heads,d_v=self.d_v)
        
        
        q=self.rope(q, token_positions) 
        k=self.rope(k, token_positions) 

        seq_len=x.shape[-2]
        i = torch.arange(seq_len,device=x.device).reshape(-1,1)
        j = torch.arange(seq_len,device=x.device).reshape(1,-1)
        boolean_mask= i>=j
        attention=scaled_dot_product_attention(q,k,v,boolean_mask)
        attention=rearrange(attention ,"... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        attention=einsum(self.weight_o,attention,"d_model d_model_v ,... d_model_v -> ... d_model")
        return attention
    

# Deliverable： 将因果 multi-head self-attention 实现为一个 torch.nn.Module。你的
# 实现应至少接受以下参数：
# d_model: int Transformer block 输入的维度。
# num_heads: int multi-head self-attention 中使用的 head 数量。
# 遵循 A. Vaswani et al. [8]，设置 dk = dv =
# dmodel
# h 。





# 一个 Transformer block
# 包含两个 sub-layers：一个用于 multihead self attention，另一个用于 SwiGLU feed-forward
# network。在每个 sub-layer 中，我们首先执行 RMSNorm，然后执行主操作（MHA/FF），最
# 后加上 residual connection。
class TransformerBlock(torch.nn.Module):
    def __init__(self,d_model: int ,num_heads: int,d_ff: int ,max_seq_len: int,theta: float,
                 dtype=None,device=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
# d_model: int Transformer block 输入的维度。
# num_heads: int multi-head self-attention 中使用的 head 数量。
# d_ff: int position-wise feed-forward inner layer 的维度。
        self.d_model=d_model
        self.attention=MultiHeadSelfAttentionWithRoPE(d_model,num_heads,theta,max_seq_len,
                 dtype=dtype,device=device)

        self.ffn=Positionwise_feedforward(d_model,d_ff,device=device,dtype=dtype)

        self.norm1=RMSNorm(d_model, eps = 1e-5, device=device, dtype=dtype)
# 需要两个独立的 RMSNorm
        self.norm2=RMSNorm(d_model, eps = 1e-5, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor,token_positions=None) -> torch.Tensor:
        x=self.attention(self.norm1(x),token_positions)+x
        return self.ffn(self.norm2(x))+x



# 我还是不太理解为什么kv cache和普通训练不一样
# 但是 KV cache 推理时，假设前面已经生成了 128 个 token。
# 现在你只把新 token 送进模型：


class TransformerLM(torch.nn.Module):
    def __init__(self,d_model: int ,num_heads: int,d_ff: int ,theta: float,
                 vocab_size: int,context_length: int, num_layers: int,
                 dtype=None,device=None):
        super().__init__()
#         vocab_size: int 词表大小，用于确定 token embedding matrix 的维度。
# context_length: int 最大上下文长度，用于确定 RoPE sin 和 cos buffer 的维度。
# num_layers: int 要使用的 Transformer blocks 数量。
        self.embedding=Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.blocks = torch.nn.ModuleList([
    TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
        theta=theta,
        device=device,
        dtype=dtype,
    ) for _ in range(num_layers)])
        # def forward(self, x: torch.Tensor,token_positions=None) -> torch.Tensor:
        self.ln_final=RMSNorm(d_model, eps = 1e-5, device=device, dtype=dtype)
        self.lm_head=Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids shape: (batch_size, seq_len)
    
# 训练语言模型时最常见：直接切成固定长度 block，，，，对短文本做 padding，，但 padding 通常还需要 attention mask，避免模型关注 padding token。
# 而是进模型前会被整理成一样长。
        x=self.embedding(token_ids)

        for block in self.blocks:
            x=block(x)       #token_positions故意不传，会默认计算，如果kv cache需要单独设计MultiHeadSelfAttention.forward

        x=self.ln_final(x)
        logits = self.lm_head(x)
        return logits




##     __init__:
#     创建 embedding
#     创建 num_layers 个 TransformerBlock
#     创建 final RMSNorm
#     创建 lm_head

# forward:
#     输入 token ids
#     得到 token embeddings
#     构造 token_positions
#     依次过所有 TransformerBlock
#     final norm
#     lm_head 得到 logits



# 这个作业里的 Transformer LM 是 decoder-only Transformer。






