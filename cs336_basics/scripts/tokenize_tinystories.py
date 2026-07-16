import torch
import numpy as np
from cs336_basics.bpe_increase import run_train_bpe_increase
from cs336_basics.Tokenizer import Tokenizer
    # 现在你可以把所有东西组合起来：获得训练好的 BPE tokenizer、对 training dataset 进行
# tokenization，并将其送入你编写的 training loop。

# 你现在的位置
# 你已经完成了 BPE 训练，并且有 TinyStories 10K tokenizer artifact：

# artifacts/tinystories_bpe_increase_vocab_10000.pkl
# artifacts/tinystories_bpe_increase_merges_10000.pkl
#from_files 做完之后，应该等价于你手动写：Tokenizer(vocab, merges, special_tokens)

vocab_filepath="artifacts/tinystories_bpe_increase_vocab_10000.pkl"
merges_filepath="artifacts/tinystories_bpe_increase_merges_10000.pkl"


def main():
    tokenizer=Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])

    def tokenize_file(tokenizer, input_path, output_path, vocab_size):
        count = 0
        # 为什么要先数？因为如果你要创建一个 .npy 数组，需要提前知道 shape：
        with open(input_path, "r") as f:
            for token_id in tokenizer.encode_iterable(f):# 文本文件对象本身就是 iterable。
                count += 1
        # 然后第二遍再真正写入：
        arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(count,))

        i = 0
        with open(input_path, "r") as f:
            for token_id in tokenizer.encode_iterable(f):
                arr[i] = token_id
                i += 1

        assert i == count
        arr.flush()

        loaded = np.load(output_path, mmap_mode="r")
        print(output_path, loaded.shape, loaded.dtype, loaded.min(), loaded.max())
        assert loaded.max() < vocab_size





    valid_in="data/TinyStoriesV2-GPT4-valid.txt"
    valid_out="data/val.npy"

    train_in="data/TinyStoriesV2-GPT4-train.txt"
    train_out="data/train.npy"


    tokenize_file(tokenizer, valid_in, valid_out, 10000)
    tokenize_file(tokenizer, train_in, train_out, 10000)



# 4. 保存后检查
# 每处理完一个文件，打印：

# 输出路径
# token 数量
# dtype
# 最小 token id
# 最大 token id

if __name__ == "__main__":
    main()
# cd /home/hapi/cs336/assignment1-basics
# uv run python -m cs336_basics.scripts.tokenize_tinystories