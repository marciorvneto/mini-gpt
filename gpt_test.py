from data_loader import TokenDataLoader
from model import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)

loader = TokenDataLoader(
    "data/wiki_tokens",
    split="train",
    batch_size=4,
    context_length=config.context_length,
)

x, y = loader.next_batch()

logits, loss = model(x, y)

print("x:", x.shape)
print("logits:", logits.shape)
print("loss:", loss.item())
