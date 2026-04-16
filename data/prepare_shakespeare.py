import os
import requests
import tiktoken
import numpy as np

class TinyShakespearePreparer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.input_file_path = os.path.join(self.data_dir, 'input.txt')
        self.train_bin = os.path.join(self.data_dir, 'train.bin')
        self.val_bin = os.path.join(self.data_dir, 'val.bin')
        self.enc = tiktoken.get_encoding("gpt2")
        self.data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

    def clean_existing_data(self):
        """Always delete existing binary files before starting."""
        os.makedirs(self.data_dir, exist_ok=True)
        for filepath in [self.train_bin, self.val_bin]:
            if os.path.exists(filepath):
                print(f"🗑️ Deleting old data file: {filepath}")
                os.remove(filepath)

    def process(self):
        """Main method to download, tokenize, and save Tiny Shakespeare."""
        # 0. Clean old data
        self.clean_existing_data()

        # 1. Download data
        if not os.path.exists(self.input_file_path):
            print("🌍 Downloading Tiny Shakespeare dataset...")
            response = requests.get(self.data_url)
            response.raise_for_status() # Ensure the download was successful
            with open(self.input_file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

        with open(self.input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()

        # 2. Tokenize using GPT-2 encoding
        print(f"Length of dataset in characters: {len(data)}")
        train_data = self.enc.encode(data)
        print(f"Total tokens: {len(train_data)}")

        # 3. Split Train/Val (90/10)
        n = int(0.9 * len(train_data))
        train_ids = train_data[:n]
        val_ids = train_data[n:]

        print(f"Train tokens: {len(train_ids)}")
        print(f"Val tokens: {len(val_ids)}")

        # 4. Save to binary files (uint16 is enough for gpt2 vocab size ~50k)
        print("💾 Saving to binary files...")
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        
        train_ids.tofile(self.train_bin)
        val_ids.tofile(self.val_bin)

        print("✅ Data preparation complete.")

if __name__ == '__main__':
    # Initialize the class and run the processor
    # We set data_dir='data' so it matches the directory your train.py is looking for
    preparer = TinyShakespearePreparer(data_dir='data')
    preparer.process()