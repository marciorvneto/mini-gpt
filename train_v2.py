from pathlib import Path
import argparse
import math
import os
import time

import torch
from tokenizers import Tokenizer

from data_loader import TokenDataLoader
from model import GPT, GPTConfig

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--resume",
    type=Path,
    default=None,
)

parser.add_argument(
    "--micro-batch-size",
    type=int,
    default=2,
)

parser.add_argument(
    "--grad-accum-steps",
    type=int,
    default=8,
)

parser.add_argument(
    "--stop-after",
    type=int,
    default=None,
    help="Stop after N optimizer steps. Useful for benchmarking.",
)

args = parser.parse_args()


# ---------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------

RESUME_PATH = args.resume

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_ROOT = Path(
    os.environ.get(
        "MINIGPT_DATA_ROOT",
        "data",
    )
)

RUN_ROOT = Path(
    os.environ.get(
        "MINIGPT_RUN_ROOT",
        ".",
    )
)

TRAIN_DATA_DIR = (
    DATA_ROOT / "wiki_tokens_v2"
)

VAL_DATA_DIR = (
    DATA_ROOT / "wiki_tokens_v2"
)

TOKENIZER_PATH = Path(
    "data/tokenizer.json"
)

CHECKPOINT_DIR = (
    RUN_ROOT / "checkpoints_v2"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------

MAX_STEPS = 50_000

MICRO_BATCH_SIZE = (
    args.micro_batch_size
)

GRAD_ACCUM_STEPS = (
    args.grad_accum_steps
)

# Effective tokens per optimizer update:
#
# 2 * 8 * 512 = 8192 tokens
#
# 50,000 optimizer steps ~= 409.6M token predictions.

MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 1_000

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 500
EVAL_BATCHES = 20
EVAL_BATCH_SIZE = 2

SAMPLE_INTERVAL = 1_000
CHECKPOINT_INTERVAL = 5_000


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")
print(f"Data root: {DATA_ROOT}")
print(f"Run root:  {RUN_ROOT}")

print(
    f"Micro batch: {MICRO_BATCH_SIZE}"
)

print(
    f"Grad accum:  {GRAD_ACCUM_STEPS}"
)

print(
    f"Effective sequences/update: "
    f"{MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}"
)

if device == "cuda":

    props = torch.cuda.get_device_properties(0)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{props.total_memory / 1024**3:.1f} GB"
    )


# ---------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------

def get_lr(step):

    # Linear warmup
    if step < WARMUP_STEPS:
        return (
            MAX_LR
            * (step + 1)
            / WARMUP_STEPS
        )

    # Cosine decay
    ratio = (
        (step - WARMUP_STEPS)
        / (MAX_STEPS - WARMUP_STEPS)
    )

    ratio = min(
        max(ratio, 0.0),
        1.0,
    )

    coeff = 0.5 * (
        1.0
        + math.cos(math.pi * ratio)
    )

    return (
        MIN_LR
        + coeff
        * (MAX_LR - MIN_LR)
    )


# ---------------------------------------------------------------------
# Resume / model config
# ---------------------------------------------------------------------

checkpoint = None

if RESUME_PATH is not None:

    if not RESUME_PATH.exists():
        raise RuntimeError(
            f"Checkpoint does not exist: "
            f"{RESUME_PATH}"
        )

    print(
        f"Loading checkpoint: {RESUME_PATH}"
    )

    checkpoint = torch.load(
        RESUME_PATH,
        map_location=device,
        weights_only=False,
    )

    # Use the exact model config stored
    # in the checkpoint.
    config = checkpoint["config"]

else:

    # MiniGPT Base v2
    config = GPTConfig(
        vocab_size=16_384,
        context_length=512,
        n_layers=12,
        n_heads=8,
        d_model=512,
        d_ff=2048,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

model = GPT(config).to(device)

n_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters: {n_params:,}")
print(config)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

train_loader = TokenDataLoader(
    TRAIN_DATA_DIR,
    split="train",
    batch_size=MICRO_BATCH_SIZE,
    context_length=config.context_length,
    seed=42,
)


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------

decay_params = []
no_decay_params = []

for p in model.parameters():

    if not p.requires_grad:
        continue

    if p.dim() >= 2:
        decay_params.append(p)

    else:
        no_decay_params.append(p)


optimizer = torch.optim.AdamW(
    [
        {
            "params": decay_params,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ],
    lr=MAX_LR,
    betas=(0.9, 0.95),
)


# ---------------------------------------------------------------------
# Restore checkpoint
# ---------------------------------------------------------------------

start_step = 0

if checkpoint is not None:

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    start_step = (
        checkpoint["step"] + 1
    )


    print(
        f"Resuming from step "
        f"{start_step:,}"
    )

if args.stop_after is None:
    end_step = MAX_STEPS
else:
    end_step = min(
        MAX_STEPS,
        start_step
        + args.stop_after
        - 1,
    )

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    # Fresh loader every evaluation.
    #
    # Therefore every checkpoint is evaluated
    # on the same deterministic sequence of
    # validation windows.
    val_loader = TokenDataLoader(
        VAL_DATA_DIR,
        split="val",
        batch_size=EVAL_BATCH_SIZE,
        context_length=config.context_length,
        seed=1234,
    )

    losses = []

    for _ in range(EVAL_BATCHES):

        x, y = val_loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        losses.append(
            loss.item()
        )

    model.train()

    return (
        sum(losses)
        / len(losses)
    )


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

eot_id = tokenizer.token_to_id(
    "<|endoftext|>"
)


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
):

    model.eval()

    ids = tokenizer.encode(
        prompt
    ).ids

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for _ in range(max_new_tokens):

        x_cond = x[
            :,
            -config.context_length:
        ]

        logits, _ = model(x_cond)

        logits = (
            logits[:, -1, :]
            / temperature
        )

        if top_k is not None:

            k = min(
                top_k,
                logits.size(-1),
            )

            values, _ = torch.topk(
                logits,
                k=k,
            )

            cutoff = (
                values[:, -1]
                .unsqueeze(-1)
            )

            logits = logits.masked_fill(
                logits < cutoff,
                float("-inf"),
            )

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        x = torch.cat(
            (x, next_token),
            dim=1,
        )

        if (
            eot_id is not None
            and next_token.item() == eot_id
        ):
            break

    model.train()

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=False,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

model.train()

tokens_per_step = (
    MICRO_BATCH_SIZE
    * GRAD_ACCUM_STEPS
    * config.context_length
)

print(
    f"Tokens / optimizer step: "
    f"{tokens_per_step:,}"
)

print(
    f"Target token presentations: "
    f"{MAX_STEPS * tokens_per_step:,}"
)

t0 = time.perf_counter()


for step in range(
    start_step,
    end_step + 1,
):

    # --------------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------------

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


    # --------------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulated_loss = 0.0

    for _ in range(
        GRAD_ACCUM_STEPS
    ):

        x, y = (
            train_loader.next_batch()
        )

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        # Divide before backward so accumulated
        # gradients correspond to the mean over
        # the effective batch.
        scaled_loss = (
            loss
            / GRAD_ACCUM_STEPS
        )

        scaled_loss.backward()

        accumulated_loss += (
            loss.item()
            / GRAD_ACCUM_STEPS
        )


    # --------------------------------------------------------------
    # Gradient clipping / update
    # --------------------------------------------------------------

    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )
    )

    optimizer.step()


    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------

    if step % 20 == 0:

        elapsed = (
            time.perf_counter()
            - t0
        )

        tokens_seen = (
            (step + 1)
            * tokens_per_step
        )

        steps_this_run = (
            step - start_step + 1
        )

        run_tokens = (
            steps_this_run
            * tokens_per_step
        )

        tokens_per_sec = (
            run_tokens / elapsed
        )

        print(
            f"step {step:5d} | "
            f"train {accumulated_loss:.4f} | "
            f"grad {grad_norm.item():.3f} | "
            f"lr {lr:.2e} | "
            f"tokens {tokens_seen:,} | "
            f"tok/s {tokens_per_sec:,.0f} | "
            f"time {elapsed:.1f}s"
        )


    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if step % EVAL_INTERVAL == 0:

        val_loss = estimate_loss()

        print(
            f">>> step {step:5d} | "
            f"val loss {val_loss:.4f}"
        )


    # --------------------------------------------------------------
    # Sample
    # --------------------------------------------------------------

    if step % SAMPLE_INTERVAL == 0:

        sample = generate(
            "The history of mathematics",
            max_new_tokens=100,
        )

        print()
        print("=" * 80)
        print(sample)
        print("=" * 80)
        print()


    # --------------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------------

    if (
        step > 0
        and step % CHECKPOINT_INTERVAL == 0
    ):

        path = (
            CHECKPOINT_DIR
            / f"step_{step:05d}.pt"
        )

        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "lr": lr,
                "tokens_seen": (
                    (step + 1)
                    * tokens_per_step
                ),
            },
            path,
        )

        print(
            f">>> saved {path}"
        )
