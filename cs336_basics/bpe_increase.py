

import regex as re


byte_vocab = [bytes([i]) for i in range(256)]
#<|endoftext|>
#说明：这一版是进行增量pair更新，并基于并行大文本的chunk进行处理

def collect_pretoken_counts(chunk,pretoken_counts,PAT):
    '利用chunk来多次更新预分词频率字典pretoken_counts'
    #第一步pretoken_counts
    # pretoken_counts={}

    for match in re.finditer(PAT,chunk):
        #re.finditer(...) 返回的是一个迭代器，里面每个元素是 match：而不是字符串
        
        #pretoken = match.group()**********************
        # pretoken=tuple(bytes([x]) for x in match.group().encode('utf-8'))

        pretoken=tuple(byte_vocab[x] for x in match.group().encode("utf-8"))
        #bytes(97)      # 长度为 97 的零字节
        #bytes([97])    # 一个字节：b"a"
        if pretoken not in pretoken_counts:
            pretoken_counts[pretoken]=1
        else:
            pretoken_counts[pretoken]+=1
    return pretoken_counts





def transfer_pretoken_counts(pretoken_counts):
    '初始化:构造增量更新需要维护的四个字典'
    id_to_token={}
    id_to_count={}
    pair_to_pretoken_ids={}
    pair_dict={}
    i=0

    for key, value in pretoken_counts.items():#字典直接迭代时，迭代的是 key
        id_to_token[i]=key
        id_to_count[i]=value

        for j in range(len(key)-1):
            # pair=tuple(key[i],key[i+1])报错。tuple() 这个函数只接收 一个可迭代对象
            pair = (key[j], key[j + 1])
            if pair not in pair_dict:
                pair_dict[pair]=value
            else:
                pair_dict[pair]+=value
            if pair not in pair_to_pretoken_ids:
                pair_to_pretoken_ids[pair]=set()
            pair_to_pretoken_ids[pair].add(i)
        i+=1
    

    return id_to_token,id_to_count,pair_dict,pair_to_pretoken_ids







def get_pair_counts_increase(id_to_token,id_to_count,pair_dict,best_key,pair_to_pretoken_ids):
    '增量:pair_dict统计相邻对的频率的字典'
    #best_key输入的是将要合并的pair对
    #得到需要合并的的pretoken的id集合
    id_set=pair_to_pretoken_ids[best_key].copy()#得到需要更新的id，后续以此开展

    new_id_to_token = id_to_token.copy()
    for id in id_set:#对需要更新的id
        pretoken=id_to_token[id]
        #不要原地替换 + 删除。更好的方式是：从左到右扫描旧序列，构造一个新序列。更适合先用 list：
        new_key=[]
        i=0#i需要初始化
        while i <= len(pretoken)-2:
            new_key.append(pretoken[i])
            if pretoken[i]==best_key[0]:
                if pretoken[i+1]==best_key[1]:
                    new_key.pop()
                    new_key.append(pretoken[i]+pretoken[i+1])
                    i+=1

            i+=1
        if i==len(pretoken)-1:#注意索引，如果到了边界，且没有超过边界，就可以添加最后的元素
            new_key.append(pretoken[i])
        new_key = tuple(new_key)


        new_id_to_token[id]=new_key#构造一个新字典


    #best_key输入的是将要合并的pair对
    #得到需要合并的的pretoken的id集合
    for id in id_set:#遍历包含这个 pair 的 pretoken_id 集合
        pretoken=id_to_token[id]#取出id对应的pretoken，继续像之前的遍历每个byte
        value=id_to_count[id]#取出id对应的pretoken的记数
        pair_set=set()
        for i in range(len(pretoken)-1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_set.add(pair)
            assert pair in pair_dict
            assert pair_dict[pair] >= value
            pair_dict[pair]=pair_dict.get(pair,0)-value
            if pair_dict[pair] == 0:
                del pair_dict[pair]
            
        for pair in pair_set:
            assert pair in pair_to_pretoken_ids
            assert id in pair_to_pretoken_ids[pair]
            pair_to_pretoken_ids[pair].discard(id)
            if not pair_to_pretoken_ids[pair]:
                    del pair_to_pretoken_ids[pair]


        pretoken=new_id_to_token[id]
        for i in range(len(pretoken)-1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_dict[pair]=pair_dict.get(pair,0)+value
            if pair not in pair_to_pretoken_ids:
                pair_to_pretoken_ids[pair] = set()

            pair_to_pretoken_ids[pair].add(id)


    return new_id_to_token,pair_dict,pair_to_pretoken_ids






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








def merge_dicts(dicts):
    '用来合并多个字典,累加相同key的value'
    result = {}

    for d in dicts:
        for k, v in d.items():
            result[k] = result.get(k, 0) + v#

    return result


#调用库方法
# from collections import Counter

# def merge_dicts(dicts):
#     counter = Counter()

#     for d in dicts:
#         counter.update(d)

#     return dict(counter)



def worker(task):    
    input_path, boundary_pair, pattern, special_tokens=task
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:#read binary不是r

        # 下面是串行写法；实际训练时，可以把每个 start/end 区间分发给不同进程并行处理。
        # 每个区间都以 special token 为边界，因此可以独立做 pre-tokenization 和计数。
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        start, end=boundary_pair
        f.seek(start)
        chunk_big = f.read(end - start).decode("utf-8")
        # 对当前 chunk 做 pre-tokenization，并保存每个 pre-token 的出现次数。
        #从文件里读取当前 chunk 的字节内容，然后解码成字符串


        # with open(input_path, "r", encoding="utf-8") as f:
        #     text = f.read()
        if special_tokens==[]:#对空 pattern 做 split 不合适。***********
            chunks=[chunk_big]#如果 special_tokens=[]，也应该返回列表。
        else:
            chunks=re.split(pattern, chunk_big)

        # chunks=split_on_special_tokens(input_path,special_tokens)
        pretoken_counts={}

        for chunk in chunks:
            if chunk=="":
                continue#return 0直接结束整个函数，并返回 0。#break后面的 chunk 都不会处理了。
            #第一步得到pretoken_counts:合并不同chunk的pretoken_counts
            pretoken_counts=collect_pretoken_counts(chunk,pretoken_counts,PAT)
        return pretoken_counts











#b"<|endoftext|>"写死以这个token切分大的chunk_big
#适合TinyStories / OpenWebText 这类用 <|endoftext|> 分文档的数据
# num_processes = 4写死

def run_train_bpe_increase(input_path,vocab_size,special_tokens,num_processes=8):
    special_tokens = list(dict.fromkeys(special_tokens))#去重（不用集合）
    #因为这里去重后还要保留原来的顺序。
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for byte vocabulary and special tokens")
 #**********************256 个基础 byte token 的 id      bytes([i])注意[i]
    # vocab = {}
    # for i in range(256):
    #     vocab[i] = bytes([i])##special_tokens[i].encode("utf-8")
    vocab_256={i:bytes([i]) for i in range(256)}
    vocab_special={256+i:special_tokens[i].encode("utf-8") for i in range(len(special_tokens))}
    # vocab=vocab_256+vocab_special#字典不能用 + 合并**********************
    vocab = vocab_256 | vocab_special

    # pretoken_counts_list=[]
    pattern="|".join(re.escape(token) for token in special_tokens)
## 用法示例
    with open(input_path, "rb") as f:#read binary不是r
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    

    from concurrent.futures import ProcessPoolExecutor
    # if __name__ == "__main__":
  
    # 2. 准备任务数据
        # tasks=(input_path, zip(boundaries[:-1], boundaries[1:]), pattern, special_tokens)
    tasks = [(input_path, boundary_pair, pattern, special_tokens)
    for boundary_pair in zip(boundaries[:-1], boundaries[1:])]
        

    with ProcessPoolExecutor(num_processes) as executor:
        results = executor.map(worker, tasks)
        # worker(input_path, tasks, pattern, special_tokens)
    


    
    results = list(results)#
    pretoken_counts=merge_dicts(results)#合并方式不能用|
# #全局变量容易导致 bug，，，Python 标准库里的 collections.Counter


    
    merges=[]
    # vocab_size=262
    # special_tokens=[]
    max_step=vocab_size - 256 - len(special_tokens)


    id_to_token,id_to_count,pair_dict,pair_to_pretoken_ids=transfer_pretoken_counts(pretoken_counts)#得到初始的

    
    merges_count=0
    while merges_count<max_step and pair_dict!={}:
        #当前词表大小 < vocab_size 且 还有 pair 可以合并:
        #第二步迭代pretoken_counts得到pair_dict
        # pair_dict=get_pair_counts(pretoken_counts)

        # if pair_dict=={}:
        #     break
        
        #第三步或者最大的pair
        max_value = max(pair_dict.values())
        candidate_keys = [key for key, value in pair_dict.items() if value == max_value]
        best_key = max(candidate_keys)
        merges.append(best_key)
        vocab[len(vocab)]=best_key[0] + best_key[1]


        #第四步得到合并后的pretoken_counts
        id_to_token,pair_dict,pair_to_pretoken_ids=get_pair_counts_increase(id_to_token,
        id_to_count,pair_dict,best_key,pair_to_pretoken_ids)

        merges_count+=1
                
                            
    return vocab, merges       
        








    
        
