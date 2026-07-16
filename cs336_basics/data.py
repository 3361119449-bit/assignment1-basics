import torch

import numpy as np

# 5.1 Data Loader 数据加载器
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





