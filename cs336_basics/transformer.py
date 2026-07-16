import torch

from cs336_basics.attention import MultiHeadSelfAttentionWithRoPE
from cs336_basics.layers import Embedding, Linear, Positionwise_feedforward, RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        dtype=None,
        device=None,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        *args,
        **kwargs,
    ):
        # d_model: 模型维度。
        # num_heads: attention head 数量。
        # d_ff: 前馈网络中间层维度。
        # max_seq_len: 最大序列长度。
        # theta: RoPE 的 theta 值。
        # dtype: 参数数据类型。
        # device: 参数所在设备。
        super().__init__(*args, **kwargs)
        if norm_position not in {"pre", "post"}:
            raise ValueError("norm_position must be 'pre' or 'post'")

        self.d_model = d_model
        self.norm_position = norm_position
        self.attention = MultiHeadSelfAttentionWithRoPE(
            d_model,
            num_heads,
            theta,
            max_seq_len,
            dtype=dtype,
            device=device,
            use_rope=use_rope,
        )
        self.ffn = Positionwise_feedforward(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        if use_rmsnorm:
            self.norm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
            self.norm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        else:
            self.norm1 = torch.nn.Identity()
            self.norm2 = torch.nn.Identity()

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        # x: 输入张量。
        # token_positions: 可选 token 位置张量。
        if self.norm_position == "pre":
            x = self.attention(self.norm1(x), token_positions) + x
            return self.ffn(self.norm2(x)) + x

        z = self.norm1(x + self.attention(x, token_positions))
        return self.norm2(z + self.ffn(z))


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        dtype=None,
        device=None,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
    ):
        # d_model: 模型维度。
        # num_heads: attention head 数量。
        # d_ff: 前馈网络中间层维度。
        # theta: RoPE 的 theta 值。
        # vocab_size: 词表大小。
        # context_length: 最大上下文长度。
        # num_layers: Transformer block 数量。
        # dtype: 参数数据类型。
        # device: 参数所在设备。
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError("norm_position must be 'pre' or 'post'")

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                )
                for _ in range(num_layers)
            ]
        )
        if use_rmsnorm and norm_position == "pre":
            self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        else:
            self.ln_final = torch.nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: token ID 张量。
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
