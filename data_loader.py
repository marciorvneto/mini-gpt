from pathlib import Path

import numpy as np
import torch


class TokenDataLoader:
    def __init__(
        self,
        data_dir,
        split,
        batch_size,
        context_length,
        seed=42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split

        self.B = batch_size
        self.T = context_length

        self.rng = np.random.default_rng(seed)

        paths = sorted(
            self.data_dir.glob(f"{split}_*.bin")
        )

        if len(paths) == 0:
            raise RuntimeError(
                f"No shards found for split '{split}'"
            )

        self.shards = [
            np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
            )
            for path in paths
        ]

        n_tokens = sum(len(shard) for shard in self.shards)

        print(
            f"{split}: "
            f"{len(self.shards)} shards, "
            f"{n_tokens:,} tokens"
        )

    def next_batch(self):

        x = torch.empty(
            (self.B, self.T),
            dtype=torch.long,
        )

        y = torch.empty(
            (self.B, self.T),
            dtype=torch.long,
        )

        for b in range(self.B):

            # Choose a shard.
            shard_idx = self.rng.integers(
                0,
                len(self.shards)
            )

            shard = self.shards[shard_idx]

            # Need T+1 tokens so that we have
            # both input and next-token targets.
            start = self.rng.integers(
                0,
                len(shard) - self.T
            )

            tokens = np.asarray(
                shard[start:start + self.T + 1],
                dtype=np.int64,
            )

            tokens = torch.from_numpy(tokens)

            x[b] = tokens[:-1]
            y[b] = tokens[1:]

        return x, y
