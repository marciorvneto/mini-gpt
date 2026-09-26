from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CORPUS_PATH    = Path("data/wiki_tokenizer_corpus.txt")
TOKENIZER_PATH = Path("data/tokenizer.json")

VOCAB_SIZE = 16_384

def main():
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
            )
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=[
                "<|endoftext|>",
                ],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
            )

    print("Training tokenizer")

    tokenizer.train(
            [str(CORPUS_PATH)],
            trainer=trainer
            )

    tokenizer.save(str(TOKENIZER_PATH))

    print()
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved to: {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()
