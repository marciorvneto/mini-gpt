import json
from pathlib import Path

import torch
from tokenizers import Tokenizer


class SFTDataLoader:
    def __init__(
        self,
        path,
        tokenizer_path,
        batch_size,
        context_length,
        seed=42,
        shuffle=True,
    ):
        self.path = Path(path)
        self.B = batch_size
        self.T = context_length
        self.shuffle = shuffle

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.eot_id = self.tokenizer.token_to_id(
            "<|endoftext|>"
        )

        if self.eot_id is None:
            raise RuntimeError(
                "Tokenizer has no <|endoftext|> token"
            )

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.examples = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.order = torch.arange(
            len(self.examples)
        )

        if self.shuffle:
            self.order = self.order[
                torch.randperm(
                    len(self.order),
                    generator=self.generator,
                )
            ]

        self.pos = 0

        print(
            f"{self.path.name}: "
            f"{len(self.examples):,} examples"
        )

    def _reset(self):
        self.pos = 0

        if self.shuffle:
            self.order = self.order[
                torch.randperm(
                    len(self.order),
                    generator=self.generator,
                )
            ]

    def _encode_example(self, example):

        prompt_ids = self.tokenizer.encode(
            example["prompt"]
        ).ids

        response_ids = self.tokenizer.encode(
            example["response"]
        ).ids

        # We need room for:
        #
        # prompt + response + EOT
        #
        # and x/y shifting needs T+1 total tokens.

        max_total_tokens = self.T + 1

        # Always preserve the prompt.
        if len(prompt_ids) >= max_total_tokens:
            raise RuntimeError(
                "Prompt is too long for context window"
            )

        # Leave one token for EOT.
        max_response_tokens = (
            max_total_tokens
            - len(prompt_ids)
            - 1
        )

        response_ids = response_ids[
            :max_response_tokens
        ]

        tokens = (
            prompt_ids
            + response_ids
            + [self.eot_id]
        )

        tokens = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        x = tokens[:-1].clone()
        y = tokens[1:].clone()

        # ----------------------------------------------------------
        # Mask prompt loss
        # ----------------------------------------------------------
        #
        # y[i] is the token AFTER x[i].
        #
        # If the prompt has P tokens:
        #
        # x[P-1] = final prompt token
        # y[P-1] = first response token
        #
        # So we ignore y positions 0 ... P-2.
        #

        prompt_length = len(prompt_ids)

        if prompt_length > 1:
            y[:prompt_length - 1] = -100

        # ----------------------------------------------------------
        # Pad to exactly T positions
        # ----------------------------------------------------------

        n = len(x)

        x_padded = torch.full(
            (self.T,),
            self.eot_id,
            dtype=torch.long,
        )

        y_padded = torch.full(
            (self.T,),
            -100,
            dtype=torch.long,
        )

        x_padded[:n] = x
        y_padded[:n] = y

        return x_padded, y_padded

    def next_batch(self):

        if self.pos + self.B > len(self.examples):
            self._reset()

        indices = self.order[
            self.pos:self.pos + self.B
        ]

        self.pos += self.B

        xs = []
        ys = []

        for idx in indices.tolist():
            x, y = self._encode_example(
                self.examples[idx]
            )

            xs.append(x)
            ys.append(y)

        return (
            torch.stack(xs),
            torch.stack(ys),
        )
