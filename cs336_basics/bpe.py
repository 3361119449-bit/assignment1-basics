#<|endoftext|>
def split_on_special_tokens(input_path,special_tokens):

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens==[]:#对空 pattern 做 split 不合适。***********
        return [text]#如果 special_tokens=[]，也应该返回列表。
    pattern="|".join(re.escape(token) for token in special_tokens)
    chunks=re.split(pattern, text)
    return chunks


def collect_pretoken_counts(chunk,pretoken_counts):
    '利用chunk来多次更新预分词频率字典pretoken_counts'
    #第一步pretoken_counts
    # pretoken_counts={}
    for match in re.finditer(PAT,chunk):
        #re.finditer(...) 返回的是一个迭代器，里面每个元素是 match：而不是字符串
        
        #pretoken = match.group()**********************
        pretoken=tuple(bytes([x]) for x in match.group().encode('utf-8'))
        #bytes(97)      # 长度为 97 的零字节
        #bytes([97])    # 一个字节：b"a"
        if pretoken not in pretoken_counts:
            pretoken_counts[pretoken]=1
        else:
            pretoken_counts[pretoken]+=1
    return pretoken_counts


def get_pair_counts(pretoken_counts):
    'pair_dict统计相邻对的频率的字典'
    #第二步迭代pretoken_counts得到pair_dict
    pair_dict={}
    for key, value in pretoken_counts.items():#字典直接迭代时，迭代的是 key
        for i in range(len(key)-1):
            # pair=tuple(key[i],key[i+1])报错。tuple() 这个函数只接收 一个可迭代对象
            pair = (key[i], key[i + 1])
            if pair not in pair_dict:
                pair_dict[pair]=value
            else:
                pair_dict[pair]+=value
    return pair_dict


def apply_merge_to_pretoken_counts(pretoken_counts,best_key):
    '接收pair_dict,得到新的合并merge后的pretoken_counts'
    new_pretoken_counts={}
    for key, value in pretoken_counts.items():
        #不要原地替换 + 删除。更好的方式是：从左到右扫描旧序列，构造一个新序列。更适合先用 list：
        new_key=[]
        i=0#i需要初始化
        while i <= len(key)-2:
            new_key.append(key[i])
            if key[i]==best_key[0]:
                if key[i+1]==best_key[1]:
                    new_key.pop()
                    new_key.append(key[i]+key[i+1])
                    i+=1

            i+=1
        if i==len(key)-1:#注意索引，如果到了边界，且没有超过边界，就可以添加最后的元素
            new_key.append(key[i])
        new_key = tuple(new_key)

        if new_key not in new_pretoken_counts:
            new_pretoken_counts[new_key]=value#构造一个新字典
        else:
            new_pretoken_counts[new_key]+=value
    return new_pretoken_counts



def run_train_bpe(input_path,vocab_size,special_tokens):
    pecial_tokens = list(dict.fromkeys(special_tokens))#去重（不用集合）
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

    chunks=split_on_special_tokens(input_path,special_tokens)
    pretoken_counts={}
    pair_dict={}
    merges=[]
    for chunk in chunks:
        if chunk=="":
            continue#return 0直接结束整个函数，并返回 0。#break后面的 chunk 都不会处理了。
        #第一步得到pretoken_counts:合并不同chunk的pretoken_counts
        pretoken_counts=collect_pretoken_counts(chunk,pretoken_counts)
    
    # vocab_size=262
    # special_tokens=[]
    max_step=vocab_size - 256 - len(special_tokens)

    merges_count=0
    while merges_count<max_step:
        #当前词表大小 < vocab_size 且 还有 pair 可以合并:
        #第二步迭代pretoken_counts得到pair_dict
        pair_dict=get_pair_counts(pretoken_counts)

        if pair_dict=={}:
            break

        #第三步或者最大的pair
        max_value = max(pair_dict.values())
        candidate_keys = [key for key, value in pair_dict.items() if value == max_value]
        best_key = max(candidate_keys)
        merges.append(best_key)
        vocab[len(vocab)]=best_key[0] + best_key[1]


        #第四步得到合并后的pretoken_counts

        new_pretoken_counts=apply_merge_to_pretoken_counts(pretoken_counts,best_key)

        pretoken_counts = new_pretoken_counts

        merges_count+=1

                
                            
    return vocab, merges       
        
