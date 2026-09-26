from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CHECKPOINT_PATH = Path(
    "checkpoints_sft_v2/wiki_chat_17m_v2.pt"
)

TOKENIZER_PATH = Path(
    "data/tokenizer.json"
)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
MAX_NEW_TOKENS = 150


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
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

eot_id = tokenizer.token_to_id(
    "<|endoftext|>"
)

if eot_id is None:
    raise RuntimeError(
        "Tokenizer does not contain <|endoftext|>"
    )


# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
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

model.eval()

print(
    f"Loaded {CHECKPOINT_PATH.name}"
)

print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

@torch.no_grad()
def generate(
    prompt,
    temperature=0.7,
    top_k=40,
    max_new_tokens=150,
):
    ids = tokenizer.encode(prompt).ids

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    generated = []

    for _ in range(max_new_tokens):

        # Keep only the context window.
        x_cond = x[
            :,
            -config.context_length:
        ]

        logits, _ = model(x_cond)

        logits = logits[:, -1, :]

        # Greedy mode
        if temperature <= 0:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        else:
            logits = logits / temperature

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

        token_id = next_token.item()

        if token_id == eot_id:
            break

        generated.append(token_id)

        x = torch.cat(
            (x, next_token),
            dim=1,
        )

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    temperature = DEFAULT_TEMPERATURE
    top_k = DEFAULT_TOP_K

    # Experimental conversation history.
    #
    # Our model was trained on single-turn examples, so don't expect
    # robust multi-turn behavior yet.
    history = ""

    print()
    print("=" * 72)
    print("Mini-GPT Chat")
    print("=" * 72)

    print("""
Commands:
  /quit
  /reset
  /temp 0.7
  /topk 40
  /greedy
""")

    while True:

        try:
            user_input = input("You> ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # ----------------------------------------------------------
        # Commands
        # ----------------------------------------------------------

        if user_input == "/quit":
            break

        if user_input == "/reset":
            history = ""
            print("Conversation reset.")
            continue

        if user_input == "/greedy":
            temperature = 0.0
            print("Greedy decoding enabled.")
            continue

        if user_input.startswith("/temp "):
            try:
                temperature = float(
                    user_input.split(
                        maxsplit=1
                    )[1]
                )

                print(
                    f"Temperature = "
                    f"{temperature}"
                )

            except ValueError:
                print("Invalid temperature.")

            continue

        if user_input.startswith("/topk "):
            try:
                top_k = int(
                    user_input.split(
                        maxsplit=1
                    )[1]
                )

                print(f"top-k = {top_k}")

            except ValueError:
                print("Invalid top-k.")

            continue

        # ----------------------------------------------------------
        # Construct prompt
        # ----------------------------------------------------------

        prompt = (
            history
            + f"User: {user_input}\n"
            + "Assistant:"
        )

        response = generate(
            prompt,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        print()
        print(f"MiniGPT> {response}")
        print()

        # Preserve the exchange.
        history += (
            f"User: {user_input}\n"
            f"Assistant:{response}"
            f"<|endoftext|>"
        )


if __name__ == "__main__":
    main()
