from tokenizers import Tokenizer

from prepare_sft_data_v2 import (
    FRACTIONS,
    GREETINGS,
    build_splits,
    extract_birth,
    extract_kind,
    extract_lifespan,
    is_bad_article,
    is_person,
    possible_examples,
    quotas,
    split_sentences,
)


TOKENIZER = Tokenizer.from_file("data/tokenizer.json")


def article(article_id, title, text):
    return {
        "id": article_id,
        "title": title,
        "text": text,
    }


STANDERT = article(
    "55615914",
    "Frederick Standert",
    "Frederick Standert (c. 1705–1785) was a British politician "
    "who sat in the House of Commons between 1769 and 1780.\n\n"
    "Standert was educated at Merchant Taylors' School. "
    "THIS LATER PARAGRAPH MUST NOT LEAK.",
)

HARRIS = article(
    "21928939",
    "Andrew Harris (Canadian football)",
    "Andrew Harris (born April 24, 1987) is a Canadian professional "
    "Canadian football running back for the Toronto Argonauts. "
    "Harris is a four-time Grey Cup champion and a league rushing leader. "
    "He played for the BC Lions before joining the Winnipeg Blue Bombers "
    "and retired as one of the league's most productive runners.\n\n"
    "THIS LATER PARAGRAPH MUST NOT LEAK.",
)

SCHOOL = article(
    "1535",
    "School No. 1535, Moscow",
    "School № 1535 is a secondary school for students of years 7-11 "
    "in the Khamovniki District of Moscow, Russia. "
    "It was founded in 1975 and teaches the later grades. "
    "The school is known for its mathematics classes.\n\n"
    "THIS LATER PARAGRAPH MUST NOT LEAK.",
)

WAR = article(
    "1914",
    "World War I",
    "World War I (1914–1918) was a global war originating in Europe. "
    "It involved the major powers of the period. "
    "The fighting stopped in November 1918 after years of conflict.\n\n"
    "Later sections.",
)


def test_sentence_splits():
    standert = split_sentences(
        "Frederick Standert (c. 1705–1785) was a politician. "
        "He sat in Parliament."
    )
    assert len(standert) == 2
    assert "1705" in standert[0]
    assert standert[1].startswith("He sat")

    martin = split_sentences(
        "Charles R. Martin was a chemist. He taught in Florida."
    )
    assert len(martin) == 2
    assert martin[0].startswith("Charles R. Martin")

    grant = split_sentences(
        "U.S. Grant was a general. He later served as president."
    )
    assert len(grant) == 2
    assert grant[0].startswith("U.S. Grant")

    school = split_sentences(
        "No. 1535 is a school. It is in Moscow."
    )
    assert len(school) == 2
    assert school[0].startswith("No. 1535")

    doctor = split_sentences(
        "Dr. King spoke in public. The crowd listened quietly."
    )
    assert len(doctor) == 2


def test_filters_and_person():
    assert is_bad_article(
        "List of cats",
        "The following is a list of cats.",
    )
    assert is_bad_article(
        "Mercury (disambiguation)",
        "Mercury may refer to several things.",
    )
    assert not is_bad_article(
        "Frederick Standert",
        "Frederick Standert was a politician.",
    )

    assert is_person(
        "Frederick Standert (c. 1705–1785) was a British politician."
    )
    assert is_person(
        "Andrew Harris (born April 24, 1987) is a running back."
    )
    assert not is_person(
        "World War I (1914–1918) was a global war."
    )
    assert not is_person(
        "School № 1535 is a secondary school in Moscow."
    )


def test_extractors():
    standert = (
        "Frederick Standert (c. 1705–1785) was a British politician "
        "who sat in the House of Commons between 1769 and 1780."
    )
    span = extract_lifespan(standert)
    assert span["birth_year"] == 1705
    assert span["death_year"] == 1785
    assert span["birth_circa"]

    birth = extract_birth(
        "Andrew Harris (born April 24, 1987) is a running back."
    )
    assert birth["display"] == "April 24, 1987"
    assert birth["year"] == 1987
    assert not birth["circa"]

    circa = extract_birth("Jim Coles (born c. 1980) is a producer.")
    assert circa["circa"]
    assert circa["year"] == 1980

    # "BC" in a team name is not the era marker.
    bc_lions = extract_birth(
        "Andrew Harris (born April 24, 1987) played for the BC Lions."
    )
    assert bc_lions["year"] == 1987

    # A war's year range is not a lifespan.
    assert extract_lifespan(
        "World War I (1914–1918) was a global war."
    ) is None

    kind = extract_kind(standert)
    assert kind["verb"] == "was"
    assert kind["phrase"] == "a British politician"


def test_examples_are_not_all_biographies():
    standert = possible_examples(STANDERT, TOKENIZER)
    harris = possible_examples(HARRIS, TOKENIZER)
    school = possible_examples(SCHOOL, TOKENIZER)
    war = possible_examples(WAR, TOKENIZER)

    # One sentence is not an overview and not a summary.
    assert "overview" not in standert
    assert "summary" not in standert
    assert "definition" in standert
    assert "House of Commons" in standert["definition"]["response"]
    assert "MUST NOT LEAK" not in standert["definition"]["response"]

    # The lifespan is a short fact, not the lead copied back.
    fact = standert["factual"]
    assert "When was" in fact["prompt"] or "When did" in fact["prompt"]
    assert "House of Commons" not in fact["response"]
    assert "1705" in fact["response"] or "1785" in fact["response"]

    assert "overview" in harris
    assert "Grey Cup" in harris["overview"]["response"]
    assert "Grey Cup" not in harris["definition"]["response"]
    assert harris["definition"]["response"].startswith("Andrew Harris")
    assert "April 24, 1987" in harris["factual"]["response"] or (
        "1987" in harris["factual"]["response"]
    )
    assert "born" in harris["factual"]["prompt"].lower()

    assert "1975" in school["factual"]["response"]
    assert "founded" in school["factual"]["prompt"]
    assert "Where is" not in school["factual"]["prompt"]

    assert "die" not in war["factual"]["prompt"].lower()
    assert "born" not in war["factual"]["prompt"].lower()

    for group in (standert, harris, school, war):
        for example in group.values():
            assert example["prompt"].startswith("User: ")
            assert example["prompt"].endswith("\nAssistant:")
            assert "MUST NOT LEAK" not in example["prompt"]
            assert "MUST NOT LEAK" not in example["response"]
            assert "Where is" not in example["prompt"]
            assert example["response"].strip()


def test_handwritten_seeds():
    expected = [
        ("User: Hi!\nAssistant:", "Hello! How can I help you?"),
        ("User: Hello.\nAssistant:", "Hi! What would you like to talk about?"),
        ("User: How are you?\nAssistant:", "I'm doing well. How can I help?"),
        ("User: Thanks!\nAssistant:", "You're welcome!"),
        ("User: Bye!\nAssistant:", "Goodbye!"),
    ]

    for prompt, response in expected:
        assert any(
            item["prompt"] == prompt and item["response"] == response
            for item in GREETINGS
        )


def test_quotas_sum():
    train = quotas(40_000)
    val = quotas(2_000)

    assert sum(train.values()) == 40_000
    assert sum(val.values()) == 2_000
    assert train["overview"] == 8_000
    assert train["greeting"] == 4_000
    assert set(train) == set(FRACTIONS)


def _rich(i):
    title = f"Example Person {i}"

    return article(
        str(i),
        title,
        (
            f"{title} (born April 2, 1950) is an American writer and "
            f"historian who lived for many years in a quiet town in Ohio. "
            f"{title} published three books about rivers, markets, and "
            f"the small towns of the Midwest. "
            f"Students in several classes still read those books today, "
            f"because the chapters describe how daily life changed. "
            f"The later chapters also describe the roads between towns.\n\n"
            f"THIS LATER PARAGRAPH MUST NOT LEAK {i}."
        ),
    )


def test_build_splits_fills_each_behavior():
    targets_train = {task: 4 for task in FRACTIONS}
    targets_val = {task: 2 for task in FRACTIONS}

    result = build_splits(
        (_rich(i) for i in range(1_000, 1_400)),
        TOKENIZER,
        targets_train,
        targets_val,
        max_articles=400,
    )

    assert result["train_counts"] == targets_train
    assert result["val_counts"] == targets_val

    prompts = [example["prompt"] for example in result["train"]]
    assert any("Tell me about" in prompt or "Who was" in prompt or "Who is" in prompt or "overview" in prompt or "Describe" in prompt for prompt in prompts)
    assert any("What is " in prompt or "Explain " in prompt or "Define " in prompt for prompt in prompts)
    assert any("Summarize" in prompt or "summary" in prompt for prompt in prompts)
    assert any(prompt.startswith("User: Hi!") or "Hello." in prompt or "How are you?" in prompt for prompt in prompts)
    assert not all("Tell me about" in prompt for prompt in prompts)

    for example in result["train"] + result["val"]:
        assert "MUST NOT LEAK" not in example["prompt"]
        assert "MUST NOT LEAK" not in example["response"]

    tasks = {example["task"] for example in result["train"]}
    assert tasks == set(FRACTIONS)


def main():
    test_sentence_splits()
    test_filters_and_person()
    test_extractors()
    test_examples_are_not_all_biographies()
    test_handwritten_seeds()
    test_quotas_sum()
    test_build_splits_fills_each_behavior()
    print("all passed")


if __name__ == "__main__":
    main()
