from pathlib import Path
import math
import time

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from sft_loader import SFTDataLoader


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

BASE_CHECKPOINT = Path(
    "checkpoints/wiki_base_17m.pt"
)

OUTPUT_DIR = Path("checkpoints_sft_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4
MAX_STEPS = 5_000

MAX_LR = 1e-5
MIN_LR = 2e-6

WARMUP_STEPS = 200

WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 50

SAMPLE_INTERVAL = 100
CHECKPOINT_INTERVAL = 250


# ---------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------

def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    ratio = (
        (step - WARMUP_STEPS)
        / (MAX_STEPS - WARMUP_STEPS)
    )

    ratio = min(max(ratio, 0.0), 1.0)

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * ratio)
    )

    return MIN_LR + coeff * (
        MAX_LR - MIN_LR
    )


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

if device == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ---------------------------------------------------------------------
# Load base model
# ---------------------------------------------------------------------

checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

config = checkpoint.get(
    "config",
    GPTConfig(),
)

model = GPT(config).to(device)

model.load_state_dict(
    checkpoint["model"]
)

print(
    f"Loaded base model from step "
    f"{checkpoint['step']:,}"
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

train_loader = SFTDataLoader(
    "data/sft_v2/train.jsonl",
    "data/tokenizer.json",
    batch_size=BATCH_SIZE,
    context_length=config.context_length,
    seed=42,
    shuffle=True,
)

val_loader = SFTDataLoader(
    "data/sft_v2/val.jsonl",
    "data/tokenizer.json",
    batch_size=BATCH_SIZE,
    context_length=config.context_length,
    seed=1234,
    shuffle=False,
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
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    loader = SFTDataLoader(
        "data/sft_v2/val.jsonl",
        "data/tokenizer.json",
        batch_size=BATCH_SIZE,
        context_length=config.context_length,
        seed=1234,
        shuffle=False,
    )

    total_nll = 0.0
    total_tokens = 0

    for _ in range(EVAL_BATCHES):

        x, y = loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        n = (y != -100).sum().item()

        total_nll += loss.item() * n
        total_tokens += n

    model.train()

    return total_nll / total_tokens


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    "data/tokenizer.json"
)


@torch.no_grad()
def generate(
    instruction,
    max_new_tokens=120,
    temperature=0.7,
    top_k=40,
):

    model.eval()

    prompt = (
        f"User: {instruction}\n"
        f"Assistant:"
    )

    ids = tokenizer.encode(prompt).ids

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

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:

            values, _ = torch.topk(
                logits,
                k=min(
                    top_k,
                    logits.size(-1),
                ),
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

        # Stop when assistant emits EOT.
        if next_token.item() == tokenizer.token_to_id(
            "<|endoftext|>"
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

t0 = time.perf_counter()

for step in range(MAX_STEPS + 1):

    lr = get_lr(step)

    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = train_loader.next_batch()

    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad(
        set_to_none=True
    )

    _, loss = model(x, y)

    loss.backward()

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
            time.perf_counter() - t0
        )

        print(
            f"step {step:5d} | "
            f"train {loss.item():.4f} | "
            f"grad {grad_norm.item():.3f} | "
            f"lr {lr:.2e} | "
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

        print()
        print("=" * 80)

        prompts = [
            "Hi!",
            "What is 2 + 2?",
            "Tell me about Albert Einstein.",
            "Give three facts about water.",
        ]

        for prompt in prompts:
            print(generate(prompt))
            print("-" * 80)

        print()

    # --------------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------------

    if (
        step > 0
        and step % CHECKPOINT_INTERVAL == 0
    ):

        path = (
            OUTPUT_DIR
            / f"sft_step_{step:05d}.pt"
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

        print(
            f">>> saved {path}"
        )
