import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

class OpenWebTextPreparer:
    def __init__(self, data_dir='data', num_proc=8):
        self.data_dir = data_dir
        self.num_proc = num_proc
        self.enc = tiktoken.get_encoding("gpt2")
        
        # Paths for binary files
        self.train_bin = os.path.join(self.data_dir, 'train.bin')
        self.val_bin = os.path.join(self.data_dir, 'val.bin')

    def clean_existing_data(self):
        """Always delete existing binary files before starting."""
        os.makedirs(self.data_dir, exist_ok=True)
        for filepath in[self.train_bin, self.val_bin]:
            if os.path.exists(filepath):
                print(f"🗑️ Deleting old data file: {filepath}")
                os.remove(filepath)

    def process(self):
        """Main method to download, tokenize, and save OpenWebText."""
        self.clean_existing_data()

        print("🌍 Downloading/Loading OpenWebText (This may take a while)...")
        # load_dataset caches the data locally. num_proc uses multiple CPU cores
        dataset = load_dataset("openwebtext", num_proc=self.num_proc)

        print("🔀 Splitting into Train and Validation sets...")
        # Take a small validation split (0.05% of 9 million docs is ~4500 docs for val)
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test') # rename test to val

        # We need an End Of Text token to separate different web documents
        eot = self.enc._special_tokens['<|endoftext|>']

        def tokenize_batch(example):
            # encode_ordinary ignores special tokens in text, then we append the EOT token
            ids = self.enc.encode_ordinary(example['text'])
            ids.append(eot)
            return {'ids': ids, 'len': len(ids)}

        print("🤖 Tokenizing dataset using GPT-2 BPE...")
        tokenized = split_dataset.map(
            tokenize_batch,
            remove_columns=['text'],
            desc="Tokenizing",
            num_proc=self.num_proc,
        )

        # Write to binary files
        for split, dset in tokenized.items():
            # Calculate total length of all tokens in this split
            total_len = np.sum(dset['len'], dtype=np.uint64)
            filename = self.train_bin if split == 'train' else self.val_bin
            
            print(f"💾 Writing {split} data to {filename} ({total_len:,} tokens)...")
            
            # Use memmap to write directly to disk without loading everything in RAM
            dtype = np.uint16 # uint16 can hold up to 65,535 (GPT-2 vocab is 50,257)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_len,))
            
            # Write sequentially to disk
            idx = 0
            for example in tqdm(dset, desc=f"Writing {split}.bin"):
                batch = example['ids']
                arr[idx : idx + len(batch)] = batch
                idx += len(batch)
            
            arr.flush() # Ensure everything is written to disk

        print("✅ OpenWebText Data preparation complete!")

if __name__ == '__main__':
    # Lower this from 16 to 4 so it doesn't overwhelm your container's internet connection
    preparer = OpenWebTextPreparer(data_dir='data', num_proc=4) 
    preparer.process()