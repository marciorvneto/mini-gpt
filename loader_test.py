import torch

from data_loader import TokenDataLoader


loader = TokenDataLoader(
    data_dir="data/wiki_tokens",
    split="train",
    batch_size=4,
    context_length=512,
)

x, y = loader.next_batch()

print("x:", x.shape)
print("y:", y.shape)

print()
print(x[0, :20])
print(y[0, :20])

assert torch.equal(
    x[:, 1:],
    y[:, :-1],
)

print("\nShift check passed.")
