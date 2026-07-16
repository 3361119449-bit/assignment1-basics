import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,#已经打开的二进制文件对象
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可以独立统计的若干块。

    如果多个边界最后落在同一个 special token 附近，返回的块数可能会少于期望值。
    这样做是为了避免把同一个文档切断后，让 pre-tokenization 跨过文档边界。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件总大小，单位是字节。
    file.seek(0, os.SEEK_END)#把文件读取位置移动到文件末尾
    file_size = file.tell()#tell() 会返回当前文件指针的位置。因为现在指针在文件末尾，
    #所以这个位置就是：文件总字节数
    file.seek(0)#把文件指针移动回文件开头。因为后面还要从文件里读取内容，所以不能一直停在末尾。


    chunk_size = file_size // desired_num_chunks#每个 chunk 大概应该有多少字节。整数除法，只保留整数部分


    # 先按均匀间隔猜测每个 chunk 的边界位置。
    # 每个 chunk 从前一个边界开始，不包含后一个边界本身。
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前多读 4KB，寻找下一个 special token。

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从猜测的边界位置开始查找。
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块用于搜索边界。

            # 如果已经到达文件末尾，则这个边界应当放在文件末尾。
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在当前小块中寻找 special token。
            # 找到后，把边界移动到 special token 的起始位置，保证 chunk 不会切在文档中间。
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去重并排序；由于边界可能重合，最终 chunk 数可能少于 desired_num_chunks。
    return sorted(set(chunk_boundaries))


## 用法示例
with open(..., "rb") as f:#read binary不是r
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 下面是串行写法；实际训练时，可以把每个 start/end 区间分发给不同进程并行处理。
    # 每个区间都以 special token 为边界，因此可以独立做 pre-tokenization 和计数。
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 对当前 chunk 做 pre-tokenization，并保存每个 pre-token 的出现次数。
        #从文件里读取当前 chunk 的字节内容，然后解码成字符串
