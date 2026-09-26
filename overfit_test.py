import torch

from model import GPT, GPTConfig
from data_loader import TokenDataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

config = GPTConfig()
model = GPT(config).to(device)

# Use a smaller sequence for this debugging experiment.
loader = TokenDataLoader(
    "data/wiki_tokens",
    split="train",
    batch_size=2,
    context_length=128,
)

# IMPORTANT: get ONE batch and reuse it forever.
x, y = loader.next_batch()

x = x.to(device)
y = y.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)

model.train()

for step in range(201):

    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(x, y)

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if step % 10 == 0:
        print(
            f"step {step:4d} | "
            f"loss {loss.item():.4f} | "
            f"grad norm {grad_norm.item():.4f}"
        )
