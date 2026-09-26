from pathlib import Path
import math

import torch
from tokenizers import Tokenizer

from data_loader import TokenDataLoader
from model import GPT, GPTConfig


CHECKPOINT_PATH = Path("checkpoints/wiki_base_17m.pt")
TOKENIZER_PATH = Path("data/tokenizer.json")

BATCH_SIZE = 4
EVAL_BATCHES = 100

PROMPTS = [
    "The history of mathematics",
    "Albert Einstein was",
    "The theory of relativity",
    "The United States",
    "In mathematics, a function",
    "South Africa",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

config = checkpoint.get("config", GPTConfig())

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Checkpoint step: {checkpoint['step']:,}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters:      {n_params:,}")


# ---------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)


# ---------------------------------------------------------------------
# Fixed validation evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_validation():

    # Recreating this with the same seed means we sample
    # exactly the same validation windows every run.
    loader = TokenDataLoader(
        "data/wiki_tokens",
        split="val",
        batch_size=BATCH_SIZE,
        context_length=config.context_length,
        seed=12345,
    )

    total_loss = 0.0

    for _ in range(EVAL_BATCHES):
        x, y = loader.next_batch()

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        total_loss += loss.item()

    mean_loss = total_loss / EVAL_BATCHES
    perplexity = math.exp(mean_loss)

    return mean_loss, perplexity


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=120,
    temperature=0.8,
    top_k=None,
    seed=42,
):
    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    for _ in range(max_new_tokens):

        x_cond = x[:, -config.context_length:]

        logits, _ = model(x_cond)

        # We only need the prediction after the final token.
        logits = logits[:, -1, :]

        # temperature = 0 -> greedy decoding
        if temperature == 0:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        else:
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))

                values, _ = torch.topk(
                    logits,
                    k=k,
                )

                cutoff = values[:, -1].unsqueeze(-1)

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

    return tokenizer.decode(
        x[0].tolist(),
        skip_special_tokens=False,
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

val_loss, perplexity = evaluate_validation()

print()
print("=" * 80)
print("VALIDATION")
print("=" * 80)
print(f"Loss:       {val_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")


settings = [
    ("greedy", 0.0, None),
    ("temp=0.5", 0.5, None),
    ("temp=0.8", 0.8, None),
    ("temp=0.8 top-k=40", 0.8, 40),
]


for prompt in PROMPTS:

    print()
    print("#" * 80)
    print(f"PROMPT: {prompt}")
    print("#" * 80)

    for name, temperature, top_k in settings:

        text = generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            seed=42,
        )

        print()
        print(f"--- {name} ---")
        print(text)
