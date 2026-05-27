import regex as re
from typing import Iterable, Iterator
import pickle


#如果作业没有强制要求 JSON 或 GPT-2 文件格式，pickle 是最适合你现在实现的方式。它的心智负担最小，不会被 bytes 编码问题绊住。

#读取文件流不够高效


#保存时确实用了 pickle.dump。



#类的内部vocab[len(vocab)] = token_bytes会原地修改外部的vocab
#如果 special token 不在 vocab，就 append。作业接口要求你的类本身也支持这个行为。
class Tokenizer:
# vocab: dict[int, bytes]
# merges: list[tuple[bytes, bytes]]
# special_tokens: list[str] | None = None
    def __init__(self, vocab, merges, special_tokens=None):
        self.merges=merges
        self.vocab=vocab.copy()
        if not special_tokens:
            self.special_tokens=[]
        else:
            self.special_tokens=special_tokens
        
        self.special_tokens=sorted(self.special_tokens, key=len, reverse=True)
        for token_str in self.special_tokens:
            token_bytes=token_str.encode("utf-8")#token.encode("utf-8") 已经是 bytes 了。
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)]=token_bytes
        

        self.bytes_to_id= {v: k for k, v in self.vocab.items()}#需要反转字典vocab->编码需要用
        #需要merges_rank这个字典，方便判断pair合并的优先级
        self.merge_ranks={pair:i for i,pair in enumerate(merges)}
        #pair -> 这个 pair 在 merges 里的顺序



        self.byte_vocab = [bytes([i]) for i in range(256)]
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


#         保存 vocab/merges
# 处理 special_tokens: None -> []
# 按长度从长到短排序 special_tokens##"<|endoftext|><|endoftext|>"希望优先被认作一个token而不是被优先切割为两个token
# 把缺失的 special token 追加到 vocab
# 创建 bytes_to_id
# 创建 merge_ranks



#from_files 做完之后，应该等价于你手动写：Tokenizer(vocab, merges, special_tokens)
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
#类方法，从序列化的词表和合并列表（格式与你的 BPE 训练代码输出相同）以及（可选
#的）特殊 token 列表构造并返回一个 Tokenizer。
# vocab_filepath: str
# merges_filepath: str
# special_tokens: list[str] | None = None
        with open(vocab_filepath,"rb") as f:#open(..., "rb") 是对的，因为 pickle 是二进制格式。
            vocab=pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges=pickle.load(f)
        return cls(vocab, merges, special_tokens)








    def encode(self, text: str) -> list[int]:
    #将输入文本编码为 token ID 序列。
    #训练时丢掉 special token 是对的；编码时不对。
    #在完整的 encode 里，special token 不应该返回到 pretoken 里继续参与 BPE。




        #说明：GPT-2 风格 pre-tokenization 正则，只用于普通文本，不用于 special token。
        if text=="":
            return []
        
        if self.special_tokens==[]:#对空 pattern 做 split 不合适。***********
            chunks=[text]#如果 special_tokens=[]，也应该返回列表。
        else:
            pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            chunks=re.split(pattern, text)
#说明：如果存在 special tokens，就先按 special token 切分文本，并保留 special token 本身，避免它被普通正则拆碎。

        id_list=[]
        for chunk in chunks:
            if chunk=="":
                continue
            if chunk in self.special_tokens:
#说明：special token 作为整体 token，直接查 id，不参与 pre-tokenization 和 BPE merge。
                id_list.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue

            #说明：普通文本 chunk 才会进入这里，每个 match 是一个 pre-token。
            for match in re.finditer(self.PAT,chunk):
                #这个循环运行完会得到pretoken[[b"t", b"h", b"e"],[b" ", b"c", b"a", b"t"]]
                pretoken_bytes=list(self.byte_vocab[x] for x in match.group().encode("utf-8"))
                #第一个 pretoken_bytes 'the' 初始表示为 [b't', b'h', b'e']
              
                for step in range(len(self.merges)):
                    #说明：重复查找当前 pre-token 中优先级最高的可合并 pair，直到没有 pair 能合并。
                    merge_id=len(self.merges)
                    for i in range(len(pretoken_bytes)-1):
                        pair_now=(pretoken_bytes[i],pretoken_bytes[i+1])
                        if pair_now in self.merge_ranks:
                            merge_id=min(self.merge_ranks[pair_now],merge_id)

                    
                    if merge_id==len(self.merges):
                        break    
                    pair=self.merges[merge_id]
                    new_token=[]
                    i=0#i需要初始化
                    while i <= len(pretoken_bytes)-2:
                        new_token.append(pretoken_bytes[i])
                        if pretoken_bytes[i]==pair[0]:
                            if pretoken_bytes[i+1]==pair[1]:
                                new_token.pop()
                                new_token.append(pretoken_bytes[i]+pretoken_bytes[i+1])
                                i+=1

                        i+=1
                    if i==len(pretoken_bytes)-1:#注意索引，如果到了边界，且没有超过边界，就可以添加最后的元素
                        new_token.append(pretoken_bytes[i])
                    pretoken_bytes = new_token
#说明：BPE merge 完成后，把每个 bytes token 转成最终 token id。
                for token in pretoken_bytes:
                    id_list.append(self.bytes_to_id[token])


        return id_list


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        #     给定一个字符串 iterable（例如 Python file handle），返回一个惰性地产生 token ID 的
        # generator。这是对无法直接加载到内存的大文件进行内存高效分词所必需的。
        #Memory considerations 内存考虑



        # for text in iterable:
        #     ids=self.encode(text)
        #     # 逐个 yield token id
        #     # generator 函数：yield 一个结果，然后暂停，下次需要时继续
        #     for token_id in ids:
        #         yield token_id#这个简单版有一个潜在问题：如果 iterable 的片段刚好把一个 token 切开，
        #         #结果可能和整体 encode 不完全一致。

        #一直保留最后一段可能不完整的文本，不立刻编码它

#你应该让 safe_cut 永远来自某个 regex match 的边界，比如 match.end() 或 match.start()，不要用 min 直接改它。

#safe_text = "hello <|"防止buffer的最后一段是特殊token的一部分


#
# chunk 拼进 buffer
# → 先检查 buffer 末尾是否可能是某个 special token 的前缀
# → 如果可能，就把这段尾巴留下，不送进 regex
# → 对安全部分先按完整 special token 切分
# → 普通文本部分再跑 GPT-2 regex

        def _adjust_cut_if_inside_special(self, text: str, cut: int) -> int:
            """
            如果 cut 落在某个完整 special token 内部，
            就把 cut 移到这个 special token 的开头。
            """
            new_cut = cut

            for special in self.special_tokens:
                L = len(special)

                # 只需要检查 cut 附近可能覆盖 cut 的位置
                start_min = max(0, cut - L + 1)
                start_max = min(cut, len(text) - L)

                for start in range(start_min, start_max + 1):
                    end = start + L
                    if text.startswith(special, start) and start < cut < end:
                        new_cut = min(new_cut, start)

            return new_cut


        buffer = ""
        for text in iterable:
            buffer += text

            protected_start = len(buffer)

            # 1. 保护末尾 special token 的“不完整前缀”
            for special in self.special_tokens:
                max_prefix_len = min(len(special) - 1, len(buffer))

                for prefix_len in range(max_prefix_len, 0, -1):
                    if buffer.endswith(special[:prefix_len]):
                        protected_start = min(
                            protected_start,
                            len(buffer) - prefix_len,
                        )
                        break

            search_text = buffer[:protected_start]

            # 2. 在非保护区里找 pre-token 安全边界
            prev_match = None
            safe_cut = 0

            for match in re.finditer(self.PAT, search_text):
                if prev_match is not None:
                    safe_cut = prev_match.end()
                prev_match = match

            if safe_cut <= 0:
                continue

            # 3. 关键修正：不能切在完整 special token 内部
            safe_cut = _adjust_cut_if_inside_special(buffer, safe_cut)

            if safe_cut <= 0:
                continue

            safe_text = buffer[:safe_cut]

            for token_id in self.encode(safe_text):
                yield token_id

            buffer = buffer[safe_cut:]

        # 4. 文件结束后，剩下的 buffer 可以全部处理
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id
            


#在 buffer 里找最后一个 pre-token 的起点。



                

    def decode(self, ids: list[int]) -> str:
        #将 token ID 序列解码为文本。

        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)
        # if ids==[]:
        #     return ""
        # byte_sequence=self.vocab[ids[0]]
        # for id in ids[1:]:
        #     byte_sequence=byte_sequence+self.vocab[id]
        text = byte_sequence.decode("utf-8", errors="replace")#先收集 bytes pieces，再一次性 join。
        #注意：不要对每个 token 的 bytes 单独 decode。
        return text






