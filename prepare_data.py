from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = Path("data/wiki_tokenizer_corpus.txt")
TARGET_BYTES = 128 * 1024 * 1024 # 128 MiB
SHUFFLE_BUFFER = 10_000
SEED = 42

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    wiki = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True
            )

    wiki = wiki.shuffle(
            seed=SEED,
            buffer_size=SHUFFLE_BUFFER
            )

    bytes_written = 0
    articles_written = 0

    with OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
            newline="\n"
            ) as f:
        with tqdm(
                total=TARGET_BYTES,
                unit="B",
                unit_scale=True,
                desc="Wikipedia"
                ) as pbar:
            for article in wiki:
                text = article["text"].strip()


                if not text:
                    continue

                chunk = text + "\n\n"

                encoded_size = len(chunk.encode("utf-8"))

                f.write(chunk)


                bytes_written += encoded_size
                articles_written += 1

                pbar.update(encoded_size)

                if bytes_written >= TARGET_BYTES:
                    break

        print()
        print(f"Articles: {articles_written:,}")
        print(f"Bytes:    {bytes_written:,}")
        print(f"Output:   {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
