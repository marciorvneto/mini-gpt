from pathlib import Path
import hashlib
import json

from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


TOKENIZER_PATH = Path("data/tokenizer.json")

OUTPUT_DIR = Path("data/sft")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"

TRAIN_EXAMPLES = 25_000
VAL_EXAMPLES = 1_000

MIN_RESPONSE_TOKENS = 24
MAX_RESPONSE_TOKENS = 320

SHUFFLE_BUFFER = 10_000
SEED = 42


PROMPT_TEMPLATES = [
    "Tell me about {title}.",
    "Give me a short overview of {title}.",
    "What can you tell me about {title}?",
    "Briefly explain {title}.",
]


def article_hash(article_id):
    digest = hashlib.blake2b(
        str(article_id).encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, "little")


def is_validation_article(article_id):
    # About 3.8%, close to our desired 1000 / 26000 split.
    return article_hash(article_id) % 26 == 0


def choose_prompt(article_id, title):
    h = article_hash(article_id)

    template = PROMPT_TEMPLATES[
        h % len(PROMPT_TEMPLATES)
    ]

    return template.format(title=title)


def get_intro(text):
    """
    Take the first non-empty paragraph.
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    if not paragraphs:
        return None

    return paragraphs[0]


def is_bad_article(title, intro):
    title_lower = title.lower()
    intro_lower = intro.lower()

    # Obvious list / disambiguation-like pages.
    if title_lower.startswith("list of "):
        return True

    if title_lower.endswith("(disambiguation)"):
        return True

    if "may refer to:" in intro_lower:
        return True

    if "can refer to:" in intro_lower:
        return True

    return False


def make_example(article, tokenizer):
    title = article["title"].strip()
    text = article["text"].strip()

    if not title or not text:
        return None

    intro = get_intro(text)

    if intro is None:
        return None

    if is_bad_article(title, intro):
        return None

    response_ids = tokenizer.encode(intro).ids
    n_response_tokens = len(response_ids)

    if not (
        MIN_RESPONSE_TOKENS
        <= n_response_tokens
        <= MAX_RESPONSE_TOKENS
    ):
        return None

    instruction = choose_prompt(
        article["id"],
        title,
    )

    # Notice that "User:" and "Assistant:" are just ordinary text.
    # We are NOT changing the tokenizer vocabulary.
    prompt = (
        f"User: {instruction}\n"
        f"Assistant:"
    )

    return {
        "article_id": article["id"],
        "title": title,
        "prompt": prompt,
        "response": intro,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    wiki = wiki.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER,
    )

    n_train = 0
    n_val = 0

    with (
        TRAIN_PATH.open(
            "w",
            encoding="utf-8",
        ) as train_file,
        VAL_PATH.open(
            "w",
            encoding="utf-8",
        ) as val_file,
    ):

        train_bar = tqdm(
            total=TRAIN_EXAMPLES,
            desc="train",
        )

        val_bar = tqdm(
            total=VAL_EXAMPLES,
            desc="val",
        )

        for article in wiki:

            if (
                n_train >= TRAIN_EXAMPLES
                and n_val >= VAL_EXAMPLES
            ):
                break

            example = make_example(
                article,
                tokenizer,
            )

            if example is None:
                continue

            if is_validation_article(
                article["id"]
            ):
                if n_val >= VAL_EXAMPLES:
                    continue

                val_file.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                n_val += 1
                val_bar.update(1)

            else:
                if n_train >= TRAIN_EXAMPLES:
                    continue

                train_file.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                n_train += 1
                train_bar.update(1)

        train_bar.close()
        val_bar.close()

    print()
    print(f"Train examples: {n_train:,}")
    print(f"Val examples:   {n_val:,}")

    print(f"Train file: {TRAIN_PATH}")
    print(f"Val file:   {VAL_PATH}")


if __name__ == "__main__":
    main()
