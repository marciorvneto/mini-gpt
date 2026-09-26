from pathlib import Path
import json
import math

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from sft_loader import SFTDataLoader


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

BASE_CHECKPOINT = Path("checkpoints/wiki_base_17m.pt")
SFT_DIR = Path("checkpoints_sft_v2")

VAL_PATH = Path("data/sft_v2/val.jsonl")
TOKENIZER_PATH = Path("data/tokenizer.json")

BATCH_SIZE = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------

checkpoints = [
    BASE_CHECKPOINT,
    *sorted(SFT_DIR.glob("sft_step_*.pt")),
]

print()
print("Checkpoints:")

for path in checkpoints:
    print(f"  {path}")


# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

eot_id = tokenizer.token_to_id(
    "<|endoftext|>"
)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(model, config):

    # Fresh loader every time:
    # same examples, same order.
    loader = SFTDataLoader(
        VAL_PATH,
        TOKENIZER_PATH,
        batch_size=BATCH_SIZE,
        context_length=config.context_length,
        seed=1234,
        shuffle=False,
    )

    n_examples = len(loader.examples)

    if n_examples % BATCH_SIZE != 0:
        raise RuntimeError(
            "For this evaluator, validation size must "
            "be divisible by batch size."
        )

    n_batches = n_examples // BATCH_SIZE

    total_nll = 0.0
    total_tokens = 0

    model.eval()

    for _ in range(n_batches):

        x, y = loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        # Number of response tokens actually contributing
        # to cross entropy.
        n_supervised = (
            y != -100
        ).sum().item()

        total_nll += (
            loss.item()
            * n_supervised
        )

        total_tokens += n_supervised

    mean_loss = (
        total_nll / total_tokens
    )

    perplexity = math.exp(mean_loss)

    return (
        mean_loss,
        perplexity,
        total_tokens,
    )


# ---------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    config,
    instruction,
    temperature=0.7,
    top_k=40,
    seed=42,
    max_new_tokens=120,
):

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

    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    model.eval()

    for _ in range(max_new_tokens):

        x_cond = x[
            :,
            -config.context_length:
        ]

        logits, _ = model(x_cond)

        logits = logits[:, -1, :]

        if temperature == 0:

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        else:

            logits = (
                logits / temperature
            )

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
                generator=generator,
            )

        x = torch.cat(
            (x, next_token),
            dim=1,
        )

        if next_token.item() == eot_id:
            break

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=False,
    )


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

PROMPTS = [
    "Tell me about Albert Einstein.",
    "What can you tell me about Super Blues?",
    "Briefly explain School No. 1535, Moscow.",
    "Tell me about South Africa.",
]


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

results = []

for checkpoint_path in checkpoints:

    print()
    print("=" * 80)
    print(checkpoint_path)
    print("=" * 80)

    checkpoint = torch.load(
        checkpoint_path,
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

    loss, ppl, n_tokens = (
        evaluate_checkpoint(
            model,
            config,
        )
    )

    print(
        f"SFT val loss: {loss:.4f}"
    )
    print(
        f"SFT perplexity: {ppl:.2f}"
    )
    print(
        f"Supervised tokens: {n_tokens:,}"
    )

    results.append({
        "checkpoint": checkpoint_path.name,
        "loss": loss,
        "ppl": ppl,
    })

    for prompt in PROMPTS:

        print()
        print("-" * 80)
        print(prompt)

        text = generate(
            model,
            config,
            prompt,
            temperature=0.7,
            top_k=40,
            seed=42,
        )

        print(text)

    del model

    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

results.sort(
    key=lambda r: r["loss"]
)

for result in results:

    print(
        f"{result['checkpoint']:30s} | "
        f"loss {result['loss']:.4f} | "
        f"ppl {result['ppl']:.2f}"
    )
