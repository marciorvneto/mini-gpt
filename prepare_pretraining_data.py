from pathlib import Path
import hashlib

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

TOKENIZER_PATH      = Path("data/tokenizer.json")
OUTPUT_DIR          = Path("data/wiki_tokens")

TRAIN_TARGET_TOKENS = 500_000_000
VAL_TARGET_TOKENS   = 10_000_000

SHARD_TOKENS        = 10_000_000

SHUFFLE_BUFFER      = 10_000
SEED                = 42

class ShardWriter:
    def __init__(
        self,
        output_dir,
        split,
        shard_tokens,
        target_tokens
        ):
        self.output_dir    = output_dir
        self.split         = split
        self.shard_tokens  = shard_tokens
        self.target_tokens = target_tokens

        self.buffer = np.empty(shard_tokens, dtype=np.uint16) # 2 bytes per token

        self.buffer_pos    = 0
        self.total_tokens  = 0
        self.shard_idx     = 0

    @property
    def done(self):
        return self.total_tokens >= self.target_tokens

    def write(self, tokens):
        if self.done:
            return

        tokens = np.asarray(tokens, dtype=np.uint16)

        remaining_dataset = self.target_tokens - self.total_tokens
        tokens            = tokens[:remaining_dataset]

        src_pos = 0

        while src_pos < len(tokens):
            remaining_buffer = self.shard_tokens - self.buffer_pos
            n = min(
                    remaining_buffer,
                    len(tokens) - src_pos
                    )

            self.buffer[self.buffer_pos:self.buffer_pos+n] = tokens[src_pos:src_pos+n]
            self.buffer_pos   += n
            self.total_tokens += n
            src_pos           += n

            if self.buffer_pos == self.shard_tokens:
                self.flush()

    def flush(self):
        if self.buffer_pos == 0:
            return

        path = self.output_dir / (
                f"{self.split}_{self.shard_idx:05d}.bin"
                )
        self.buffer[:self.buffer_pos].tofile(path)

        print(
            f"\nWrote {path} "
            f"({self.buffer_pos:,} tokens)"
        )

        self.buffer_pos = 0
        self.shard_idx += 1

def is_validation_article(article_id):
    digest = hashlib.blake2b(
            str(article_id).encode("utf-8"),
            digest_size=8
            ).digest()

    value = int.from_bytes(digest, "little")
    return value % 50 == 0

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(
            str(TOKENIZER_PATH)
            )

    eot_id = tokenizer.token_to_id("<|endoftext|>")

    if eot_id is None:
        raise RuntimeError("Expected <|endoftext|> to be define. Check the tokenizer.")

    
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"EOT token ID:    {eot_id}")

    wiki = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True
            )

    wiki = wiki.shuffle(
            seed=SEED,
            buffer_size=SHUFFLE_BUFFER
            )

    train_writer = ShardWriter(
            OUTPUT_DIR,
            "train",
            SHARD_TOKENS,
            TRAIN_TARGET_TOKENS
            )

    val_writer = ShardWriter(
            OUTPUT_DIR,
            "val",
            SHARD_TOKENS,
            VAL_TARGET_TOKENS
            )

    train_bar = tqdm(
            total=TRAIN_TARGET_TOKENS,
            desc="train",
            unit="tok",
            position=0
            )

    val_bar = tqdm(
            total=VAL_TARGET_TOKENS,
            desc="val",
            unit="tok",
            position=1
            )

    for article in wiki:

        if train_writer.done and val_writer.done:
            break

        text = article["text"].strip()

        if not text:
            continue

        ids = tokenizer.encode(text).ids
        ids.append(eot_id)

        if is_validation_article(article["id"]):

            if not val_writer.done:
                before = val_writer.total_tokens
                val_writer.write(ids)

                val_bar.update(val_writer.total_tokens - before)

        else:

            if not train_writer.done:
                before = train_writer.total_tokens
                train_writer.write(ids)

                train_bar.update(train_writer.total_tokens - before)

    train_writer.flush()
    val_writer.flush()

    train_bar.close()
    val_bar.close()

    print()
    print(
        f"Train tokens: {train_writer.total_tokens:,}"
        )
    print(
        f"Val tokens:   {val_writer.total_tokens:,}"
        )



if __name__ == "__main__":
    main()
