import math

import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # in_features: 输入张量的最后一维大小。
        # out_features: 输出张量的最后一维大小。
        # device: 参数所在设备。
        # dtype: 参数数据类型。
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 输入张量。
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings: 词表大小。
        # embedding_dim: embedding 向量维度。
        # device: 参数所在设备。
        # dtype: 参数数据类型。
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: token ID 张量。
        # assert token_ids.dtype == torch.long

        # num_embeddings = self.weight.shape[0]
        # embedding_dim = self.weight.shape[1]

        # assert token_ids.max().item() < num_embeddings
        # assert token_ids.min().item() >= 0

        output = self.weight[token_ids]
        # assert output.shape[-1] == embedding_dim

        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # d_model: 隐藏层维度。
        # eps: 数值稳定项。
        # device: 参数所在设备。
        # dtype: 参数数据类型。
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 输入张量。
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model
            + self.eps
        ).unsqueeze(-1)
        result = einsum(
            x / rms,
            self.weight.to(torch.float32),
            "... d_model, d_model -> ... d_model",
        )
        return result.to(in_dtype)


class Positionwise_feedforward(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        # d_model: 模型维度。
        # d_ff: 前馈网络中间层维度。
        # device: 参数所在设备。
        # dtype: 参数数据类型。
        super().__init__()
        if d_ff is None:
            self.d_ff = int(((8 / 3 * d_model) + 32) // 64 * 64)
        else:
            self.d_ff = d_ff

        self.weight_1 = torch.nn.Parameter(
            torch.empty(self.d_ff, d_model, device=device, dtype=dtype)
        )
        self.weight_2 = torch.nn.Parameter(
            torch.empty(d_model, self.d_ff, device=device, dtype=dtype)
        )
        self.weight_3 = torch.nn.Parameter(
            torch.empty(self.d_ff, d_model, device=device, dtype=dtype)
        )

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
        # x: 输入张量。
        w1x = einsum(x, self.weight_1, "... d_model, d_ff d_model -> ... d_ff")
        x = w1x * torch.sigmoid(w1x) * einsum(
            x,
            self.weight_3,
            "... d_model, d_ff d_model -> ... d_ff",
        )
        return einsum(x, self.weight_2, "... d_ff, d_model d_ff -> ... d_model")


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # theta: RoPE 的 theta 值。
        # d_k: query 和 key 向量维度。
        # max_seq_len: 最大序列长度。
        # device: buffer 所在设备。
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        positions = torch.arange(max_seq_len, device=device).float().reshape(-1, 1)
        inv_freq = 1 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )
        inv_freq = inv_freq.reshape(1, -1)
        table = inv_freq * positions
        sin_table = torch.sin(table)
        cos_table = torch.cos(table)

        self.register_buffer("sin_table", sin_table, persistent=False)
        self.register_buffer("cos_table", cos_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: 输入张量。
        # token_positions: token 位置张量。
        if self.sin_table.device != x.device or self.cos_table.device != x.device:
            raise RuntimeError(
                f"RoPE buffers are on {self.sin_table.device}, but x is on {x.device}. "
                f"Call rope.to(x.device), or construct RoPE with device=x.device."
            )

        token_positions = token_positions.to(device=x.device, dtype=torch.long)
        sin_table = self.sin_table[token_positions].to(dtype=x.dtype, device=x.device)
        cos_table = self.cos_table[token_positions].to(dtype=x.dtype, device=x.device)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        while sin_table.ndim < x_even.ndim:
            sin_table = sin_table.unsqueeze(-3)
            cos_table = cos_table.unsqueeze(-3)

        x_even_new = x_even * cos_table - x_odd * sin_table
        x_odd_new = x_even * sin_table + x_odd * cos_table

        result = torch.empty_like(x)
        result[..., 0::2] = x_even_new
        result[..., 1::2] = x_odd_new
        return result
