from pathlib import Path
import math
import time

import torch
from tokenizers import Tokenizer

from data_loader import TokenDataLoader
from model import GPT, GPTConfig


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RESUME_PATH = Path("checkpoints/step_05000.pt")

MAX_STEPS = 30_000
BATCH_SIZE = 4

MAX_LR = 3e-4
MIN_LR = 3e-5

LR_DECAY_START = 5_000
LR_DECAY_END = MAX_STEPS

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 200
EVAL_BATCHES = 20

SAMPLE_INTERVAL = 500
CHECKPOINT_INTERVAL = 1_000

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------

def get_lr(step):
    if step <= LR_DECAY_START:
        return MAX_LR

    if step >= LR_DECAY_END:
        return MIN_LR

    ratio = (
        (step - LR_DECAY_START)
        / (LR_DECAY_END - LR_DECAY_START)
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * ratio)
    )

    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------

config = GPTConfig()

model = GPT(config).to(device)

train_loader = TokenDataLoader(
    "data/wiki_tokens",
    split="train",
    batch_size=BATCH_SIZE,
    context_length=config.context_length,
    seed=42,
)

val_loader = TokenDataLoader(
    "data/wiki_tokens",
    split="val",
    batch_size=BATCH_SIZE,
    context_length=config.context_length,
    seed=1234,
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
# Resume checkpoint
# ---------------------------------------------------------------------

start_step = 0

if RESUME_PATH is not None:
    if not RESUME_PATH.exists():
        raise RuntimeError(
            f"Checkpoint does not exist: {RESUME_PATH}"
        )

    print(f"Loading checkpoint: {RESUME_PATH}")

    checkpoint = torch.load(
        RESUME_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_step = checkpoint["step"] + 1

    print(f"Resuming from step {start_step:,}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    model.eval()

    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = val_loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    "data/tokenizer.json"
)


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
):
    model.eval()

    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for _ in range(max_new_tokens):

        x_cond = x[:, -config.context_length:]

        logits, _ = model(x_cond)

        logits = logits[:, -1, :] / temperature

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

    model.train()

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=False,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

model.train()

t0 = time.perf_counter()

for step in range(
    start_step,
    MAX_STEPS + 1,
):

    # --------------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------------

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # --------------------------------------------------------------
    # Batch
    # --------------------------------------------------------------

    x, y = train_loader.next_batch()

    x = x.to(device)
    y = y.to(device)

    # --------------------------------------------------------------
    # Forward / backward
    # --------------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    _, loss = model(x, y)

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    optimizer.step()

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------

    if step % 20 == 0:
        elapsed = time.perf_counter() - t0

        tokens_seen = (
            (step + 1)
            * BATCH_SIZE
            * config.context_length
        )

        print(
            f"step {step:5d} | "
            f"train {loss.item():.4f} | "
            f"grad {grad_norm.item():.3f} | "
            f"lr {lr:.2e} | "
            f"tokens {tokens_seen:,} | "
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
    # Samples
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
    # Checkpoints
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
            },
            path,
        )

        print(f">>> saved {path}")
