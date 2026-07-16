# Assignment 1 实验结果整理

来源：本文件根据之前和 Codex 的对话记录，以及当前仓库中的 `artifacts/`、`data/`、`runs/` 日志整理。  
用途：写作业一 writeup / experiment log 时，快速定位已经完成的实验、结果文件和可写结论。

## 0. 当前主要实验配置

低资源训练配置主要使用：

| item | value |
| --- | --- |
| dataset | TinyStories |
| tokenizer vocab size | 10000 |
| model | Transformer LM |
| vocab_size | 10000 |
| context_length | 256 |
| d_model | 512 |
| num_layers | 4 |
| num_heads | 16 |
| d_ff | 1344 |
| batch_size | 32，部分 batch-size sweep 另有变化 |
| total_steps | 5000，短实验通常 500 |
| total tokens processed | `batch_size * steps * context_length = 40,960,000` |
| optimizer | AdamW |
| dtype | `torch.float32` |
| device | `cuda` |

主要日志目录：

```text
runs/
artifacts/
data/train.npy
data/val.npy
```

## 1. BPE Training on TinyStories

作业要求：

- 训练 TinyStories byte-level BPE tokenizer。
- vocab size 最大为 10000。
- 加入 special token `<|endoftext|>`。
- 序列化 vocab 和 merges。
- 报告训练时间、内存、最长 token，并评论最长 token 是否合理。

已完成结果：

| run | processes | vocab_len | merges_len | elapsed_s | longest token | longest bytes |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| TinyStories BPE | 4 | 10000 | 9743 | 146.95 | `b' accomplishment'` | 15 |
| TinyStories BPE | 8 | 10000 | 9743 | 88.41 | `b' accomplishment'` | 15 |

结果文件：

```text
artifacts/tinystories_bpe_increase_10000_summary.json
artifacts/tinystories_bpe_increase_10000_8proc_summary.json
artifacts/tinystories_bpe_increase_vocab_10000.pkl
artifacts/tinystories_bpe_increase_merges_10000.pkl
artifacts/tinystories_bpe_increase_main_10000_profile.txt
```

可写结论：

- 8 进程并行预分词版本比 4 进程版本更快，记录时间从约 `146.95s` 降到 `88.41s`。
- 最长 token 是带前导空格的英文单词片段 `b' accomplishment'`，这符合 TinyStories 中常见英文词和 GPT-style pre-tokenization 的特点。
- 当前 summary 中没有记录 peak memory；如果 writeup 要严格回答 memory，需要再用 `/usr/bin/time -v` 或 memory profiler 补一次。

## 2. Tokenizer 与 Tokenized Dataset

作业要求：

- 实现 tokenizer encode/decode。
- 支持 special tokens。
- 将 TinyStories train/dev 数据编码成 token ID 序列，后续训练使用。

已完成结果：

| file | shape | dtype | min token id | max token id |
| --- | ---: | --- | ---: | ---: |
| `data/train.npy` | `(541229347,)` | `uint16` | 9 | 9999 |
| `data/val.npy` | `(5465883,)` | `uint16` | 10 | 9999 |

相关文件：

```text
cs336_basics/Tokenizer.py
cs336_basics/scripts/tokenize_tinystories.py
data/TinyStoriesV2-GPT4-train.txt
data/TinyStoriesV2-GPT4-valid.txt
data/train.npy
data/val.npy
```

可写结论：

- 由于 vocab size 为 10000，token ID 可以用 `uint16` 保存，节省磁盘空间。
- train set 编码后约 `541M` tokens；训练实验只随机采样其中一部分 token。

## 3. Single Batch Overfit Sanity Check

对话中做过一个固定 minibatch overfit 实验，用于验证模型、loss、optimizer 和 backward 是否基本正确。

记录结果：

```text
step    1 | loss 9.208186 | grad_norm 0.815 | time 0.7s
step   20 | loss 1.170360 | grad_norm 0.987 | time 1.0s
success: loss reached 0.047779 at step 35
```

可写结论：

- 模型能在单个 minibatch 上快速过拟合到接近 0 loss，说明 forward、loss、backward、optimizer 的主路径基本工作正常。

## 4. Learning Rate Sweep

作业要求：

- 对 learning rate 做 sweep，报告 learning curves。
- 包含至少一个 divergent run。
- 分析“edge of stability”和收敛速度的关系。

短实验结果：

| lr | final step | final val_loss | final ppl | comment |
| ---: | ---: | ---: | ---: | --- |
| `1e-4` | 500 | 3.5763 | 35.74 | 太小，收敛慢 |
| `3e-4` | 500 | 2.9694 | 19.48 | 稳定但慢 |
| `1e-3` | 500 | 2.5366 | 12.64 | 稳定 |
| `3e-3` | 500 | 2.3446 | 10.43 | 短跑较好 |
| `1e-2` | 500 | 2.8837 | 17.88 | 学习率偏大，效果变差 |
| `3e-2` | 100 | 4.1148 | 61.24 | 很差，明显不稳定 |
| `3e-1` | 300 | 5.5409 | 254.90 | 接近发散/严重不稳定 |
| `1e+0` | 100 | 47.1859 | `3.11e20` | divergent run |

日志路径：

```text
runs/lr_1e-04/metrics.csv
runs/lr_3e-04/metrics.csv
runs/lr_1e-03/metrics.csv
runs/lr_3e-03/metrics.csv
runs/lr_1e-02/metrics.csv
runs/lr_3e-02/metrics.csv
runs/lr_3e-01/metrics.csv
runs/lr_1e+00/metrics.csv
```

可写结论：

- 学习率太小时收敛慢，适中时验证损失下降最快。
- 学习率继续增大后，训练并不一定立刻 NaN，但验证损失明显变差，`1.0` 出现非常大的 validation loss，可作为 divergent run。
- 当前实验中最好的短跑学习率附近是 `3e-3`，但完整 5000 step 中 `2e-3` 的 RoPE baseline 最好。

## 5. Low-Resource TinyStories Training

作业 low-resource 提示：

- 低资源配置可使用 `32 * 5000 * 256 = 40,960,000` tokens。
- 参考解在该设置下约能达到 validation loss `1.80`。

完整训练结果：

| run | lr | final val_loss | best val_loss | best step | final ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoPE baseline | `2e-3` | 1.6698 | 1.6238 | 4800 | 5.31 |
| RoPE baseline | `3e-3` | 1.6931 | 1.6639 | 4900 | 5.44 |

日志路径：

```text
runs/batch_size_sweep/bs_032_lr_2e-03TOTAL_STEPS=5000/metrics.csv
runs/batch_size_sweep/bs_032_lr_3e-03TOTAL_STEPS=5000/metrics.csv
runs/batch_size_sweep/bs_032_lr_3e-03TOTAL_STEPS=5000/checkpoint.pt
```

可写结论：

- 在 40.96M tokens 的低资源设置下，模型能达到 `~1.62-1.67` 的 best validation loss，优于作业低资源提示中的 `1.80` 示例。
- `2e-3` 在完整训练中优于 `3e-3`，说明短跑最优不一定等于长跑最优。

## 6. Batch Size Experiment

作业要求：

- batch size 从 1 增大到 GPU memory limit。
- 至少包含中间几个 batch size，例如 64、128。
- 如有必要重新优化 learning rate。
- 提供不同 batch size 的 learning curves 和几句话分析。

短跑结果，均为 `lr=3e-3`：

| batch_size | steps | tokens processed | final val_loss | final ppl |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 500 | 128,000 | 4.1969 | 66.48 |
| 4 | 500 | 512,000 | 3.4813 | 32.50 |
| 16 | 500 | 2,048,000 | 2.7547 | 15.72 |
| 32 | 500 | 4,096,000 | 2.4297 | 11.36 |
| 64 | 500 | 8,192,000 | 2.2203 | 9.21 |
| 128 | 500 | 16,384,000 | 1.9781 | 7.23 |
| 256 | 500 | 32,768,000 | 1.8039 | 6.07 |

日志路径：

```text
runs/batch_size_sweep/bs_001_lr_3e-03/metrics.csv
runs/batch_size_sweep/bs_004_lr_3e-03/metrics.csv
runs/batch_size_sweep/bs_016_lr_3e-03/metrics.csv
runs/batch_size_sweep/bs_032_lr_3e-03/metrics.csv
runs/batch_size_sweep/bs_064_lr_3e-03/metrics.csv
runs/batch_size_sweep/bs_128_lr_3e-03/metrics.csv
runs/batch_size_sweep/bs_256_lr_3e-03/metrics.csv
```

可写结论：

- 在固定 step 数下，较大 batch size 处理的 token 更多，因此 validation loss 明显更低。
- 这个对比不是完全公平的“相同 token budget”比较；若要比较 batch size 本身的统计效应，应固定 total tokens processed。
- 大 batch 通常更稳定、GPU 利用率更高，但单步更慢、显存占用更大。

## 7. Text Generation

作业要求：

- 生成至少 256 tokens，或直到第一个 `<|endoftext|>`。
- 评论 fluency。
- 至少提到两个影响输出质量的因素。

对话中保存的 sample output：

```text
Once upon a time, there was a little boy named Tim. He had a dog named Max. Tim and Max liked to play together. One day, Tim was very hungry. He wanted to find some food.
Tim and Max walked around the park. They saw a big tree. Max said, "Let's go to the tree and look for food!" Tim agreed, and they went to the tree.
Tim and Max found a big, red apple. They took a bite and it was very tasty. Tim and Max were very happy. They ate the apple and had a great day.
<|endoftext|>
```

生成参数：

```python
prompt = "Once upon a time,"
max_generated_tokens = 200
temperature = 0.8
Top_p = 0.9
context_length = 256
```

可写结论：

- 输出语法基本通顺，符合 TinyStories 的儿童故事风格，有人物、事件和结尾。
- 内容偏简单，情节有模板化倾向，这和训练 token 数较少、模型较小、采样温度/top-p 设置有关。
- `temperature` 和 `top_p` 会影响随机性；训练步数、模型大小、数据质量也会影响 fluency。

注意：

- 作业要求至少 256 tokens；上面这段可能不足 256 tokens，因为它提前生成了 `<|endoftext|>`，按题目“or until first `<|endoftext|>`”可以接受。

## 8. Remove RMSNorm Ablation

作业要求：

- 移除 RMSNorm 后训练。
- 观察 previous optimal LR 会怎样。
- 尝试更低 LR 是否能稳定。
- 提供 learning curve 和简短评论。

短跑结果：

| No RMSNorm lr | final step | final val_loss | final ppl | comment |
| ---: | ---: | ---: | ---: | --- |
| `3e-4` | 500 | 2.8720 | 17.67 | 稳定 |
| `1e-3` | 500 | 2.4507 | 11.60 | 稳定 |
| `2e-3` | 500 | 2.3791 | 10.79 | 短跑较好 |
| `3e-3` | - | - | - | 没有完整 val 记录，之前观察为发散/崩掉 |

完整训练结果：

| No RMSNorm lr | final step | final val_loss | best val_loss | best step | comment |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1e-3` | 5000 | 1.6730 | 1.6727 | 4900 | 稳定 |
| `2e-3` | 1600 | NaN | 3.3323 | 200 | 后续 NaN，发散 |

日志路径：

```text
runs/no_rmsnorm/short_lr_3e-04_500steps/metrics.csv
runs/no_rmsnorm/short_lr_1e-03_500steps/metrics.csv
runs/no_rmsnorm/short_lr_2e-03_500steps/metrics.csv
runs/no_rmsnorm/short_lr_3e-03_500steps/metrics.csv
runs/no_rmsnorm/best_lr_1e-03_5000steps/metrics.csv
runs/no_rmsnorm/best_lr_2e-03_5000steps/metrics.csv
```

可写结论：

- 移除 RMSNorm 后模型对 learning rate 更敏感。
- `2e-3` 短跑看起来不错，但长跑后发散，说明短跑稳定不代表完整训练稳定。
- 降到 `1e-3` 后可以稳定训练，但最终 loss 与 baseline 接近或略差，RMSNorm 的主要作用体现在训练稳定性和允许更激进学习率。

## 9. Post-Norm Ablation

作业要求：

- 将 pre-norm Transformer 改成 post-norm。
- 训练 post-norm 模型。
- 提供 post-norm 与 pre-norm 的 learning curve 对比。

短跑结果：

| Post-Norm lr | final val_loss | final ppl |
| ---: | ---: | ---: |
| `3e-4` | 2.9207 | 18.55 |
| `1e-3` | 2.4042 | 11.07 |
| `2e-3` | 2.3409 | 10.39 |
| `3e-3` | 2.3072 | 10.05 |

完整训练结果：

| Post-Norm lr | final val_loss | best val_loss | best step | final ppl |
| ---: | ---: | ---: | ---: | ---: |
| `3e-4` | 1.8376 | 1.8142 | 4700 | 6.28 |
| `1e-3` | 1.6939 | 1.6523 | 4700 | 5.44 |
| `2e-3` | 1.7012 | 1.6899 | 4700 | 5.48 |
| `3e-3` | 1.7202 | 1.7158 | 4900 | 5.59 |

日志路径：

```text
runs/post_norm/short_lr_3e-04_500steps/metrics.csv
runs/post_norm/short_lr_1e-03_500steps/metrics.csv
runs/post_norm/short_lr_2e-03_500steps/metrics.csv
runs/post_norm/short_lr_3e-03_500steps/metrics.csv
runs/post_norm/full_lr_3e-04_5000steps/metrics.csv
runs/post_norm/full_lr_1e-03_5000steps/metrics.csv
runs/post_norm/full_lr_2e-03_5000steps/metrics.csv
runs/post_norm/best_lr_3e-03_5000steps/metrics.csv
```

可写结论：

- Post-norm 在这些小模型/TinyStories 设置下可以稳定训练。
- Post-norm 的最佳完整训练学习率更偏向 `1e-3`，而 pre-norm baseline 的较好结果在 `2e-3` 到 `3e-3`。
- Post-norm 最优 best val loss `1.6523`，接近但略差于 RoPE pre-norm `2e-3` baseline 的 `1.6238`。

## 10. NoPE vs RoPE Ablation

作业要求：

- 实现 NoPE，即不使用任何 position embeddings / RoPE。
- 比较 RoPE 和 NoPE 的 learning curve。

NoPE 短跑结果：

| NoPE lr | final val_loss | final ppl |
| ---: | ---: | ---: |
| `3e-4` | 3.3233 | 27.75 |
| `1e-3` | 2.8133 | 16.66 |
| `2e-3` | 2.6700 | 14.44 |
| `3e-3` | 2.7180 | 15.15 |

NoPE 完整训练结果：

| model | lr | final val_loss | best val_loss | best step | final ppl |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPE | `2e-3` | 1.7496 | 1.7326 | 4800 | 5.75 |
| RoPE | `2e-3` | 1.6698 | 1.6238 | 4800 | 5.31 |
| RoPE | `3e-3` | 1.6931 | 1.6639 | 4900 | 5.44 |

日志路径：

```text
runs/nope/short_lr_3e-04_500steps/metrics.csv
runs/nope/short_lr_1e-03_500steps/metrics.csv
runs/nope/short_lr_2e-03_500steps/metrics.csv
runs/nope/short_lr_3e-03_500steps/metrics.csv
runs/nope/full_lr_2e-03_5000steps/metrics.csv
runs/batch_size_sweep/bs_032_lr_2e-03TOTAL_STEPS=5000/metrics.csv
```

可写结论：

- NoPE 可以正常训练，没有直接发散。
- 但和相同 `lr=2e-3` 的 RoPE 对比，NoPE 的 best val loss 高约 `0.109`。
- 这说明 causal decoder-only Transformer 虽然能从 mask 和上下文中间接获得一些位置信息，但显式 RoPE 仍然提升 TinyStories language modeling 性能。

## 11. 还缺或未在对话中完成的实验

这些是作业要求中和实验相关、但本对话记录里没有完整结果的部分：

| requirement | status |
| --- | --- |
| OpenWebText BPE tokenizer 训练与 TinyStories tokenizer 对比 | 未见完整结果 |
| tokenizer compression ratio：TinyStories / OpenWebText sample | 未见完整结果 |
| OpenWebText dataset tokenization | 未见完整结果 |
| SwiGLU vs SiLU feed-forward ablation | 未见完整结果 |
| OpenWebText LM main experiment | 未见完整结果 |
| leaderboard experiment | 未见完整结果，可选 |

如果 writeup 必须覆盖这些小题，需要继续补实验或明确说明未做。

## 12. 建议画图清单

建议在 notebook 中至少画这些曲线：

1. Learning rate sweep：
   - `runs/lr_1e-04`
   - `runs/lr_3e-04`
   - `runs/lr_1e-03`
   - `runs/lr_3e-03`
   - `runs/lr_1e-02`
   - `runs/lr_1e+00`

2. Batch size sweep：
   - `runs/batch_size_sweep/bs_001_lr_3e-03`
   - `runs/batch_size_sweep/bs_004_lr_3e-03`
   - `runs/batch_size_sweep/bs_016_lr_3e-03`
   - `runs/batch_size_sweep/bs_032_lr_3e-03`
   - `runs/batch_size_sweep/bs_064_lr_3e-03`
   - `runs/batch_size_sweep/bs_128_lr_3e-03`
   - `runs/batch_size_sweep/bs_256_lr_3e-03`

3. RMSNorm ablation：
   - baseline RoPE pre-norm
   - `runs/no_rmsnorm/best_lr_1e-03_5000steps`
   - optionally `runs/no_rmsnorm/best_lr_2e-03_5000steps` to show divergence

4. Post-norm ablation：
   - baseline RoPE pre-norm
   - `runs/post_norm/full_lr_1e-03_5000steps`
   - `runs/post_norm/full_lr_2e-03_5000steps`
   - `runs/post_norm/best_lr_3e-03_5000steps`

5. NoPE ablation：
   - `runs/nope/full_lr_2e-03_5000steps`
   - `runs/batch_size_sweep/bs_032_lr_2e-03TOTAL_STEPS=5000`

## 13. 简短总述

可以写进 experiment log 的总述：

> I trained a 10K byte-level BPE tokenizer on TinyStories with `<|endoftext|>` as a special token, tokenized the train/validation splits into `uint16` token arrays, and trained a small Transformer LM under a low-resource budget of 40.96M tokens. Learning-rate sweeps showed that very small learning rates converge slowly, while very large learning rates become unstable or divergent. The best low-resource RoPE baseline reached best validation loss around 1.62. Larger batch sizes improved validation loss at fixed step count because they process more tokens. Removing RMSNorm made training more learning-rate sensitive; post-norm remained trainable but needed different LR tuning; NoPE trained successfully but underperformed RoPE, showing the benefit of explicit rotary positional information.
