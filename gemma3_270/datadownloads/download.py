from datasets import load_dataset
import os
import numpy as np
from gemma3_270.pre_training.tokenizer import Tokenizer
from tqdm.auto import tqdm


data = load_dataset("roneneldan/TinyStories")



tokenizer = Tokenizer()
if not os.path.exists("train.bin"):
    tokenized = data.map(tokenizer.text_to_token,remove_columns=['text'],desc="tokenizing the splits", num_proc=8)
    print(tokenized.values())

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint64

        # map it to file on disk
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc = f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches,index=batch_idx,contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx+len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

