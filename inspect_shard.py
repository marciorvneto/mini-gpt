from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


TOKENIZER_PATH = Path("data/tokenizer.json")
SHARD_PATH = Path("data/wiki_tokens/train_00007.bin")


tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

data = np.memmap(
    SHARD_PATH,
    dtype=np.uint16,
    mode="r",
)

print(f"Tokens in shard: {len(data):,}")
print(f"Min token ID:    {data.min()}")
print(f"Max token ID:    {data.max()}")

start = 1_000_000
length = 500

tokens = data[start:start + length]

text = tokenizer.decode(
    tokens.astype(int).tolist(),
    skip_special_tokens=False,
)

print()
print("=" * 80)
print(text)
print("=" * 80)
