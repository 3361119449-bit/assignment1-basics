import math



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