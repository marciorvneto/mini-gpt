from sft_loader import SFTDataLoader


loader = SFTDataLoader(
    "data/sft/train.jsonl",
    "data/tokenizer.json",
    batch_size=2,
    context_length=512,
)

x, y = loader.next_batch()

print(x.shape)
print(y.shape)

print()
print("x:")
print(x[0, :80])

print()
print("y:")
print(y[0, :80])

print()
print(
    "Supervised tokens:",
    (y[0] != -100).sum().item()
)

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "data/tokenizer.json"
)

supervised = y[0][y[0] != -100]

print(
    tokenizer.decode(
        supervised.tolist(),
        skip_special_tokens=False,
    )
)
