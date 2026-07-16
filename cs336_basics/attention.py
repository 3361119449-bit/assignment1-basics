import math

import torch
from einops import rearrange, einsum

from cs336_basics.layers import RoPE


def softmax(x, i):
    # x: 输入张量。
    # i: 应用 softmax 的维度。
    values, _ = torch.max(x, dim=i, keepdim=True)
    exp_x = torch.exp(x - values)
    sum_x = exp_x.sum(dim=i, keepdim=True)
    return exp_x / sum_x


def scaled_dot_product_attention(queries, key, values, boolean_mask=None):
    # queries: query 张量。
    # key: key 张量。
    # values: value 张量。
    # boolean_mask: 可选 attention mask。
    d_k = key.shape[-1]
    scores = einsum(
        queries,
        key,
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k",
    ) / math.sqrt(d_k)
    if boolean_mask is not None:
        scores = scores.masked_fill(~boolean_mask, float("-inf"))

    attention = softmax(scores, -1)
    return einsum(
        attention,
        values,
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
    )


class MultiHeadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        dtype=None,
        device=None,
        use_rope: bool = True,
    ):
        # d_model: 模型维度。
        # num_heads: attention head 数量。
        # theta: RoPE 的 theta 值。
        # max_seq_len: 最大序列长度。
        # dtype: 参数数据类型。
        # device: 参数所在设备。
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.weight_q = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_model, dtype=dtype, device=device)
        )
        self.weight_k = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_model, dtype=dtype, device=device)
        )
        self.weight_v = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_model, dtype=dtype, device=device)
        )
        self.weight_o = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_model, dtype=dtype, device=device)
        )

        std = math.sqrt(1 / self.d_model)
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
        self.use_rope = use_rope
        self.rope = RoPE(theta, self.d_k, max_seq_len, device) if use_rope else None

    def forward(self, x, token_positions=None):
        # x: 输入张量。
        # token_positions: 可选 token 位置张量。
        q = einsum(
            self.weight_q,
            x,
            "d_model_q d_model, ... d_model -> ... d_model_q",
        )
        k = einsum(
            self.weight_k,
            x,
            "d_model_k d_model, ... d_model -> ... d_model_k",
        )
        v = einsum(
            self.weight_v,
            x,
            "d_model_v d_model, ... d_model -> ... d_model_v",
        )

        q = rearrange(
            q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
            d_v=self.d_v,
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)


        seq_len = x.shape[-2]
        i = torch.arange(seq_len, device=x.device).reshape(-1, 1)
        j = torch.arange(seq_len, device=x.device).reshape(1, -1)
        boolean_mask = i >= j
        attention = scaled_dot_product_attention(q, k, v, boolean_mask)
        attention = rearrange(
            attention,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
        )
        attention = einsum(
            self.weight_o,
            attention,
            "d_model d_model_v, ... d_model_v -> ... d_model",
        )
        return attention
