from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer.json")

tests = [
    "Albert Einstein was a theoretical physicist.",
    "thermodynamics",
    "electroencephalographically",
    "The pressure is 12.5 bar.",
    "A função converge uniformemente.",
    "∂u/∂t = α∇²u",
    "supercalifragilisticexpialidocious",
]

for text in tests:
    enc = tokenizer.encode(text)

    print("=" * 80)
    print("TEXT:")
    print(text)

    print("\nTOKENS:")
    print(enc.tokens)

    print("\nIDS:")
    print(enc.ids)

    print("\nN TOKENS:")
    print(len(enc.ids))

    print("\nDECODED:")
    print(tokenizer.decode(enc.ids))
