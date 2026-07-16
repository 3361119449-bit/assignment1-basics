import torch
from cs336_basics.attention import softmax

# 6 Generating text 生成文本

# prompt是字符串
# dtype = torch.long
def Decoding(model,context_length,tokenizer,prompt,max_generated_tokens,temperature=1,Top_p=None,device="cpu"):
#     • 为用户提供的 prompt 生成 completions（即接收某些 x1...t 并采样 completion，直
# 到遇到 <|endoftext|> token）。
# • 允许用户控制最大 generated tokens 数量。
# • 给定 desired temperature value，在采样前对 predicted next-token distributions
# 应用 softmax temperature scaling。
# • Top-p sampling（[A. Holtzman et al., 2020] 也称 nucleus sampling），给定用户指
# 定的 threshold value。
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prompt)
        input=torch.tensor(ids,dtype = torch.long,device=device).reshape(1,-1)
        # x = torch.cat([x, new_row], dim=0)
        
        num_tokens=0
        while num_tokens <max_generated_tokens:
            logits=model(input[:, -context_length:])#每次喂给模型时只取最后 context_length 个 token。防止太长
            if  temperature == 0:
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                p_tokens=softmax(logits[:, -1, :]/temperature,-1)[0]
                #取出最后一个词的概率分布
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