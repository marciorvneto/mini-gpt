"""
Build a mixed SFT corpus.

V1 asked only "tell me about {title}" and always answered with the
Wikipedia lead, so every input was secretly a request for a biography.
This version mixes several behaviors. Wikipedia answers are copied or
cut out of the lead. They are not paraphrased, and a missing pattern
is skipped rather than guessed.

Not generated here, on purpose:
  - "Where is X?" and other questions whose answer is buried in a clause
  - abstractive summaries ("say it in your own words")

Those need a teacher model. Summaries in this file are extractive:
the prompt contains the passage, and the answer is its first sentence.
"""

from pathlib import Path
import hashlib
import json
import random
import re

from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from model import GPTConfig


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TOKENIZER_PATH = Path("data/tokenizer.json")

OUTPUT_DIR = Path("data/sft_v2")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"

# Rough mix, not a contract. The caps exist so one behavior
# (the biography) cannot take over the file again.
TRAIN_EXAMPLES = 40_000
VAL_EXAMPLES = 2_000

FRACTIONS = {
    "overview": 0.20,
    "factual": 0.20,
    "definition": 0.15,
    "summary": 0.15,
    "greeting": 0.10,
    "transform": 0.10,
    "misc": 0.10,
}

# Two tasks per article is enough to mix behaviors without
# repeating the same lead under five prompts.
MAX_PER_ARTICLE = 2

MAX_ARTICLES = 400_000
SHUFFLE_BUFFER = 10_000
SEED = 42

# Must match the model. prompt + response + EOT has to fit,
# or the loader silently chops the answer.
CONTEXT_LENGTH = GPTConfig.context_length

OVERVIEW_MIN_TOKENS = 32
OVERVIEW_MAX_TOKENS = 320


# ---------------------------------------------------------------------
# Handwritten seeds
# ---------------------------------------------------------------------
#
# Repeated up to the greeting quota, and about half of the
# transform / misc quotas. The other half of those comes from
# Wikipedia, with a different surface form.

def _chat(task, user, response):
    return {
        "article_id": None,
        "title": None,
        "task": task,
        "source": "handwritten",
        "prompt": f"User: {user}\nAssistant:",
        "response": response,
    }


# The first five are the ones we already liked. Keep them verbatim.
GREETING_PAIRS = [
    ("Hi!", "Hello! How can I help you?"),
    ("Hello.", "Hi! What would you like to talk about?"),
    ("How are you?", "I'm doing well. How can I help?"),
    ("Thanks!", "You're welcome!"),
    ("Bye!", "Goodbye!"),
    ("Hey!", "Hey! What can I help you with?"),
    ("Hi there.", "Hello! What would you like to know?"),
    ("Hello there.", "Hi! How can I help?"),
    ("Good morning.", "Good morning! What can I do for you?"),
    ("Good afternoon.", "Good afternoon! What can I help you with?"),
    ("Good evening.", "Good evening! How can I help?"),
    ("Good night.", "Good night!"),
    ("Howdy.", "Howdy! What would you like to talk about?"),
    ("Hey there.", "Hi! What's on your mind?"),
    ("How's it going?", "It's going well. How can I help?"),
    ("How are you doing?", "I'm doing well, thanks. What do you need?"),
    ("What's up?", "Not much. What can I help you with?"),
    ("Hi, how are you?", "Hello! I'm doing well. How can I help?"),
    ("Thank you.", "You're welcome."),
    ("Thank you!", "You're welcome!"),
    ("Thanks a lot.", "Happy to help."),
    ("Thank you so much.", "You're welcome. Glad it helped."),
    ("Cheers.", "You're welcome!"),
    ("Much appreciated.", "Glad to help."),
    ("I appreciate it.", "You're welcome."),
    ("Thank you for your help.", "You're welcome."),
    ("Goodbye.", "Goodbye!"),
    ("See you.", "See you later!"),
    ("See you later.", "See you!"),
    ("Take care.", "You too. Take care!"),
    ("Bye for now.", "Goodbye!"),
    ("Later!", "Talk to you later!"),
    ("Who are you?", "I'm Mini-GPT, a small assistant."),
    ("What's your name?", "I'm Mini-GPT."),
    ("What can you do?", "I can answer questions, define things, and summarize short passages."),
    ("Can you help me?", "Yes. What do you need?"),
    ("I need some help.", "Sure. Tell me what you're working on."),
    ("Help.", "I'm here. What do you need help with?"),
    ("Are you there?", "Yes, I'm here. What can I do for you?"),
    ("Nice to meet you.", "Nice to meet you too."),
    ("Please help me.", "Of course. What do you need?"),
    ("Sorry.", "No need to apologize. How can I help?"),
]

REWRITE_PAIRS = [
    (
        "Rewrite this more simply:\nThe canine consumed its evening meal with great enthusiasm.",
        "The dog ate its dinner eagerly.",
    ),
    (
        "Say this in simpler words:\nThe automobile was traveling at a high rate of speed.",
        "The car was going very fast.",
    ),
    (
        "Rewrite this more simply:\nPrecipitation is anticipated in the afternoon.",
        "Rain is expected this afternoon.",
    ),
    (
        "Say this in simpler words:\nThe children proceeded to their residence.",
        "The children went home.",
    ),
    (
        "Rewrite this more simply:\nShe purchased a beverage at the establishment.",
        "She bought a drink at the shop.",
    ),
    (
        "Say this in simpler words:\nThe examination was exceedingly difficult.",
        "The test was very hard.",
    ),
    (
        "Rewrite this more simply:\nHe resides in a large municipality.",
        "He lives in a big city.",
    ),
    (
        "Say this in simpler words:\nThe motion picture commences at eight.",
        "The movie starts at eight.",
    ),
    (
        "Rewrite this more simply:\nUtilize this implement to inscribe your name.",
        "Use this tool to write your name.",
    ),
    (
        "Say this in simpler words:\nThe feline is reposing on the furniture.",
        "The cat is resting on the furniture.",
    ),
    (
        "Rewrite this more simply:\nWe shall commence the meeting presently.",
        "We will start the meeting soon.",
    ),
    (
        "Say this in simpler words:\nThe physician requested that he repose.",
        "The doctor asked him to rest.",
    ),
    (
        "Rewrite this more simply:\nNumerous individuals attended the gathering.",
        "Many people went to the gathering.",
    ),
    (
        "Say this in simpler words:\nThe volume is situated upon the table.",
        "The book is on the table.",
    ),
    (
        "Rewrite this more simply:\nShe was exceedingly fatigued after the journey.",
        "She was very tired after the trip.",
    ),
    (
        "Say this in simpler words:\nThe pupils completed their assignments.",
        "The students finished their homework.",
    ),
    (
        "Rewrite this more simply:\nIt is necessary for you to depart immediately.",
        "You need to leave now.",
    ),
    (
        "Say this in simpler words:\nThe weather is quite pleasant today.",
        "The weather is very nice today.",
    ),
    (
        "Rewrite this more simply:\nHe attempted to assist his companion.",
        "He tried to help his friend.",
    ),
    (
        "Say this in simpler words:\nThe edifice is extremely tall.",
        "The building is very tall.",
    ),
    (
        "Rewrite this more simply:\nThey conversed for a lengthy period.",
        "They talked for a long time.",
    ),
    (
        "Say this in simpler words:\nPlease extinguish the illumination.",
        "Please turn off the light.",
    ),
    (
        "Rewrite this more simply:\nThe infant is slumbering.",
        "The baby is sleeping.",
    ),
    (
        "Say this in simpler words:\nThis query is uncomplicated.",
        "This question is simple.",
    ),
]

THREE_FACT_PAIRS = [
    (
        "Give three facts about a triangle.",
        "- A triangle has three sides.\n- A triangle has three angles.\n- The angles of a triangle add up to 180 degrees.",
    ),
    (
        "List three facts about the week.",
        "- A week has seven days.\n- Monday and Tuesday are weekdays.\n- Saturday and Sunday are weekend days.",
    ),
    (
        "Give three facts about water.",
        "- Water is a liquid at room temperature.\n- Water freezes at 0 degrees Celsius.\n- Water boils at 100 degrees Celsius.",
    ),
    (
        "List three facts about a bicycle.",
        "- A bicycle has two wheels.\n- A bicycle is powered by pedals.\n- A bicycle is a vehicle.",
    ),
    (
        "Give three facts about the Moon.",
        "- The Moon orbits the Earth.\n- The Moon is smaller than the Earth.\n- People can see the Moon from Earth.",
    ),
    (
        "List three facts about a book.",
        "- A book has pages.\n- A book can be read.\n- A book usually has a cover.",
    ),
    (
        "Give three facts about winter.",
        "- Winter is one of the four seasons.\n- Winter comes after autumn.\n- Winter comes before spring.",
    ),
    (
        "List three facts about the Sun.",
        "- The Sun is a star.\n- The Sun gives Earth light.\n- The Sun gives Earth heat.",
    ),
    (
        "Give three facts about a square.",
        "- A square has four sides.\n- All four sides of a square are equal.\n- A square has four corners.",
    ),
]

MISC_PAIRS = [
    ("What is 2 + 2?", "4."),
    ("What is 10 minus 3?", "7."),
    ("What is 5 times 5?", "25."),
    ("What is 9 divided by 3?", "3."),
    ("What is 100 divided by 10?", "10."),
    ("Which is larger, 3 or 8?", "8."),
    ("How many days are in a week?", "Seven."),
    ("How many months are in a year?", "Twelve."),
    ("How many hours are in a day?", "24."),
    ("How many minutes are in an hour?", "60."),
    ("How many sides does a triangle have?", "Three."),
    ("How many wheels does a bicycle have?", "Two."),
    ("What comes after Monday?", "Tuesday."),
    ("What comes after Friday?", "Saturday."),
    ("What comes before April?", "March."),
    ("What is the first month of the year?", "January."),
    ("What is the opposite of hot?", "Cold."),
    ("What is the opposite of up?", "Down."),
    ("What is the opposite of day?", "Night."),
    ("What color is snow?", "White."),
    ("What color is coal?", "Black."),
    ("What do bees make?", "Honey."),
    ("What is H2O?", "Water."),
    ("How many letters are in the English alphabet?", "26."),
    ("What is the plural of cat?", "Cats."),
    ("Is ice hot or cold?", "Cold."),
    ("Name a primary color.", "Red."),
    ("How many seasons are there?", "Four."),
]

GREETINGS = [_chat("greeting", *pair) for pair in GREETING_PAIRS]
TRANSFORMS = (
    [_chat("transform", *pair) for pair in REWRITE_PAIRS]
    + [_chat("transform", *pair) for pair in THREE_FACT_PAIRS]
)
MISC_HANDWRITTEN = [_chat("misc", *pair) for pair in MISC_PAIRS]


# ---------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------

# Titles and initials. Case-sensitive so "no. He left" still splits,
# while "No. 1535" and "Charles R. Martin" do not.
_TITLE_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Ft|Gen|Col|Lt|Sgt|Capt|Rev|"
    r"Hon|Fig|No|Nos|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\."
)
_LOWER_ABBREV_RE = re.compile(
    r"\b(?:c|ca|cf|vs|etc|ed|eds|pp|al|approx)\."
)
_INITIALS_RE = re.compile(r"\b(?:[A-Z]\.){1,4}")
_SINGLE_INITIAL_RE = re.compile(r"\b[A-Z]\.")

_PERSON_RE = re.compile(
    r"\b(?:politician|actor|actress|singer|songwriter|writer|author|poet|"
    r"painter|king|queen|emperor|empress|saint|bishop|scientist|"
    r"mathematician|philosopher|composer|artist|footballer|player|"
    r"athlete|physician|lawyer|judge|soldier|explorer|inventor|"
    r"journalist|architect|prince|princess|novelist|director|musician|"
    r"professor|researcher|historian|economist|chemist|physicist|"
    r"biologist|engineer|pilot|astronaut|general|admiral|pope|"
    r"pharaoh|duke|minister|senator|governor|mayor|pianist|sculptor|"
    r"photographer|activist|scholar|coach|hermit|poet)\b",
    re.IGNORECASE,
)
_NOT_PERSON_RE = re.compile(
    r"\b(?:album|song|film|movie|series|novel|book|magazine|newspaper|"
    r"band|company|school|university|city|town|village|river|mountain|"
    r"game|episode|single|treaty|war|battle|election|software|building|"
    r"church|station|airport|species|genus|club|team|district|language|"
    r"algorithm|theory)\b",
    re.IGNORECASE,
)

_BORN_RE = re.compile(
    r"\bborn\b.{0,80}?(?P<date>"
    r"c\.\s*\d{3,4}"
    r"|ca\.\s*\d{3,4}"
    r"|\d{1,2}\s+[A-Z][a-z]+\s+\d{4}"
    r"|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"
    r"|[A-Z][a-z]+\s+\d{4}"
    r"|\d{3,4}"
    r")",
    re.IGNORECASE,
)
_DIED_RE = re.compile(
    r"\bdied\b.{0,60}?\b(?:on|in)\s+(?P<date>"
    r"\d{1,2}\s+[A-Z][a-z]+\s+\d{4}"
    r"|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"
    r"|[A-Z][a-z]+\s+\d{4}"
    r"|\d{3,4}"
    r")",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"(?P<c1>c\.|ca\.)?\s*(?P<y1>\d{3,4})\s*[–—-]\s*"
    r"(?P<c2>c\.|ca\.)?\s*(?P<y2>\d{3,4})"
)
_ORIGIN_RE = re.compile(
    r"\b(?P<verb>founded|established|created|formed|released|published|"
    r"built|opened|incorporated|launched|completed|recorded|composed)\b"
    r"(?:\s+\w+){0,4}?\s+(?:in|on)\s+(?P<date>"
    r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}"
    r"|\d{1,2}\s+[A-Z][a-z]+\s+\d{4}"
    r"|[A-Z][a-z]+\s+\d{4}"
    r"|\d{4}"
    r")",
    re.IGNORECASE,
)
_WORK_YEAR_RE = re.compile(
    r"\b(?:a|an)\s+(?P<year>(?:1[0-9]{3}|20[0-2][0-9]))\s+"
    r"(?:studio\s+|debut\s+|live\s+)?"
    r"(?:album|film|novel|book|song|single|game|series|poem|play|opera)\b",
    re.IGNORECASE,
)
_ALIAS_RE = re.compile(
    r"\b(?:also known as|better known as|professionally known as)\s+"
    r"(?P<alias>[^.;()]{2,60})",
    re.IGNORECASE,
)
_KIND_RE = re.compile(
    r"\b(?P<verb>is|was)\s+(?P<phrase>(?:a|an|the)\s+[^.;]{2,180})",
    re.IGNORECASE,
)


def article_hash(value):
    digest = hashlib.blake2b(
        str(value).encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, "little")


def is_validation_article(article_id):
    # About 5%, near the 2_000 / 42_000 split.
    return article_hash(article_id) % 20 == 0


def choose(options, key):
    return options[article_hash(key) % len(options)]


def clean_inline(text):
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_intro(text):
    """First paragraph of the article. Later sections stay out."""

    if not text or not text.strip():
        return None

    # Wikipedia blank lines sometimes contain spaces, so \n\n is not enough.
    chunks = re.split(r"\n\s*\n", text.strip())

    for chunk in chunks:
        paragraph = clean_inline(chunk)

        if paragraph:
            return paragraph

    return None


def _protect_dots(match):
    return match.group(0).replace(".", "∯")


def split_sentences(text):
    text = clean_inline(text)

    if not text:
        return []

    protected = _TITLE_ABBREV_RE.sub(_protect_dots, text)
    protected = _INITIALS_RE.sub(_protect_dots, protected)
    protected = _SINGLE_INITIAL_RE.sub(_protect_dots, protected)
    protected = _LOWER_ABBREV_RE.sub(_protect_dots, protected)

    parts = re.split(r"(?<=[.!?])\s+", protected)

    sentences = []

    for part in parts:
        sentence = part.replace("∯", ".").strip()

        if sentence:
            sentences.append(sentence)

    return sentences


def is_bad_article(title, intro):
    title_lower = title.lower().strip()
    intro_lower = intro.lower()

    if title_lower.startswith("list of "):
        return True

    if title_lower.startswith("lists of "):
        return True

    if title_lower.endswith("(disambiguation)"):
        return True

    if "may refer to" in intro_lower:
        return True

    if "can refer to" in intro_lower:
        return True

    if intro_lower.startswith("redirect"):
        return True

    return False


def is_person(sentence):
    born = re.search(r"\bborn\b", sentence, re.IGNORECASE)
    role = _PERSON_RE.search(sentence)

    if not born and not role:
        return False

    # "born" is decisive. A role word plus "album" or "school" is not.
    if _NOT_PERSON_RE.search(sentence) and not born:
        return False

    return True


def strip_parens(text):
    previous = None
    current = text

    while current != previous:
        previous = current
        current = re.sub(r"\s*\([^()]*\)", "", current)

    current = re.sub(r"\s+", " ", current)
    current = re.sub(r"\s+([,.;:!?])", r"\1", current)

    return current.strip()


def n_tokens(tokenizer, text):
    return len(tokenizer.encode(text).ids)


def fits(tokenizer, prompt, response, min_tokens, max_tokens):
    response = response.strip()

    if not response:
        return False

    n_response = n_tokens(tokenizer, response)

    if n_response < min_tokens or n_response > max_tokens:
        return False

    n_prompt = n_tokens(tokenizer, prompt)

    if n_prompt < 1 or n_prompt >= CONTEXT_LENGTH:
        return False

    # prompt + response + <|endoftext|> must fit in context_length + 1,
    # which is what the loader keeps before the x/y shift.
    if n_prompt + n_response > CONTEXT_LENGTH:
        return False

    return True


def wiki_example(article, task, instruction, response):
    return {
        "article_id": str(article["id"]),
        "title": article["title"].strip(),
        "task": task,
        "source": "wiki",
        "prompt": f"User: {instruction}\nAssistant:",
        "response": response.strip(),
    }


# ---------------------------------------------------------------------
# Fact extractors
# ---------------------------------------------------------------------
#
# Each one returns None unless the lead actually contains the fact.
# Year ranges count as a lifespan only on pages that look like a person,
# so "World War I (1914–1918)" does not become a death date.

def _year_ok(year):
    return 100 <= year <= 2023


def _mentions_bc(text):
    # "BC Lions" is a team, not a year. Only an era marker after a number counts.
    return re.search(r"\d\s*(?:BC|BCE)\b", text) is not None


def parse_when(date_text):
    text = re.sub(r"\s+", " ", date_text).strip(" .")
    # "c." has no word boundary after the period, so don't use \b there.
    circa = re.match(r"(?i)(?:c\.|ca\.|circa)(?:\s|$)", text) is not None
    years = [int(year) for year in re.findall(r"\d{3,4}", text)]

    if not years:
        return None

    year = years[-1]

    if not _year_ok(year):
        return None

    if re.search(r"\d{1,2},|\d{1,2}\s+[A-Z]", text):
        precision = "day"
    elif re.search(r"[A-Z][a-z]+\s+\d{4}", text):
        precision = "month"
    else:
        precision = "year"

    display = re.sub(
        r"(?i)^(?:c\.|ca\.|circa)\s*",
        "",
        text,
    ).strip()

    return {
        "display": display,
        "circa": circa,
        "year": year,
        "precision": precision,
    }


def extract_birth(intro):
    if _mentions_bc(intro):
        return None

    match = _BORN_RE.search(intro)

    if match is None:
        return None

    return parse_when(match.group("date"))


def extract_died(intro):
    if _mentions_bc(intro):
        return None

    match = _DIED_RE.search(intro)

    if match is None:
        return None

    return parse_when(match.group("date"))


def extract_lifespan(first_sentence):
    if not is_person(first_sentence):
        return None

    if _mentions_bc(first_sentence):
        return None

    match = _RANGE_RE.search(first_sentence)

    if match is None:
        return None

    birth_year = int(match.group("y1"))
    death_year = int(match.group("y2"))

    if not (
        _year_ok(birth_year)
        and _year_ok(death_year)
        and death_year > birth_year
    ):
        return None

    return {
        "birth_year": birth_year,
        "death_year": death_year,
        "birth_circa": bool(match.group("c1")),
        "death_circa": bool(match.group("c2")),
    }


def extract_origin(intro):
    if _mentions_bc(intro):
        return None

    match = _ORIGIN_RE.search(intro)

    if match is not None:
        when = parse_when(match.group("date"))

        if when is not None:
            return {
                "verb": match.group("verb").lower(),
                "when": when,
            }

    match = _WORK_YEAR_RE.search(intro)

    if match is None:
        return None

    when = parse_when(match.group("year"))

    if when is None:
        return None

    return {
        "verb": "released",
        "when": when,
    }


def extract_alias(intro, title):
    match = _ALIAS_RE.search(intro)

    if match is None:
        return None

    alias = re.sub(r"\s+", " ", match.group("alias")).strip(" ,")
    words = alias.split()

    if not 1 <= len(words) <= 8:
        return None

    if alias.lower() == title.lower():
        return None

    return alias


def shorten_phrase(phrase):
    phrase = re.sub(r"\s*\([^)]*\)", "", phrase)
    phrase = re.split(
        r"\s+who\b|\s+which\b|\s+that\b|,",
        phrase,
        maxsplit=1,
    )[0]
    phrase = phrase.strip(" ,;:")
    words = phrase.split()

    if 2 <= len(words) <= 14:
        return phrase

    cut = re.split(
        r"\s+(?:for|in|of|from|by|at|with|on)\s+",
        phrase,
        maxsplit=1,
    )[0].strip(" ,;:")
    cut_words = cut.split()

    if 2 <= len(cut_words) <= 14:
        return cut

    return None


def extract_kind(first_sentence):
    match = _KIND_RE.search(first_sentence)

    if match is None:
        return None

    phrase = shorten_phrase(match.group("phrase"))

    if phrase is None:
        return None

    return {
        "verb": match.group("verb").lower(),
        "phrase": phrase,
    }


def _as_sentence(phrase):
    sentence = phrase[0].upper() + phrase[1:]

    if sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def _prep_for(when):
    if when["precision"] == "day":
        return "on"

    return "in"


def _birth_responses(title, when):
    if when["circa"]:
        return [
            f"Around {when['year']}.",
            f"{title} was born around {when['year']}.",
        ]

    if when["precision"] == "year":
        return [
            f"{when['display']}.",
            f"{title} was born in {when['display']}.",
        ]

    return [
        f"{when['display']}.",
        f"{title} was born on {when['display']}.",
    ]


def _death_responses(title, when):
    if when["circa"]:
        return [
            f"Around {when['year']}.",
            f"{title} died around {when['year']}.",
        ]

    if when["precision"] == "year":
        return [
            f"{when['display']}.",
            f"{title} died in {when['display']}.",
        ]

    return [
        f"{when['display']}.",
        f"{title} died on {when['display']}.",
    ]


def _year_fact(year, circa):
    return {
        "display": str(year),
        "circa": circa,
        "year": year,
        "precision": "year",
    }


# ---------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------

def make_overview(article, title, intro, sentences, key, tokenizer):
    # One sentence is a definition, not an overview. Asking "tell me
    # about X" for those would teach the old one-line biography habit.
    if len(sentences) < 2:
        return None

    person = is_person(sentences[0])
    head = re.sub(r"\([^)]*\)", " ", sentences[0])
    has_was = re.search(r"\bwas\b", head) is not None
    has_is = re.search(r"\bis\b", head) is not None

    prompts = [
        f"Tell me about {title}.",
        f"Give me a short overview of {title}.",
        f"What can you tell me about {title}?",
        f"Briefly explain {title}.",
        f"Describe {title}.",
    ]

    if person and has_was and not has_is:
        prompts.append(f"Who was {title}?")

    if person and has_is and not has_was:
        prompts.append(f"Who is {title}?")

    instruction = choose(prompts, key + ":overview")
    example = wiki_example(article, "overview", instruction, intro)

    if not fits(
        tokenizer,
        example["prompt"],
        example["response"],
        OVERVIEW_MIN_TOKENS,
        OVERVIEW_MAX_TOKENS,
    ):
        return None

    return example


def make_definition(article, title, sentences, key, tokenizer):
    sentence = sentences[0]

    if is_person(sentence):
        prompts = [
            f"What is {title}?",
            f"Explain {title} in one sentence.",
        ]
    else:
        prompts = [
            f"What is {title}?",
            f"Define {title}.",
            f"Explain {title} in one sentence.",
        ]

    instruction = choose(prompts, key + ":definition")
    example = wiki_example(article, "definition", instruction, sentence)

    if not fits(tokenizer, example["prompt"], example["response"], 8, 90):
        return None

    return example


def make_summary(article, title, intro, sentences, key, tokenizer):
    if len(sentences) < 2:
        return None

    answer = sentences[0]

    # The answer has to be a real cut, not the passage copied back.
    if len(intro) - len(answer) < 40:
        return None

    templates = [
        "Summarize this:\n{passage}",
        "In one sentence, summarize the following:\n{passage}",
        "Give a short summary of this passage:\n{passage}",
    ]
    template = choose(templates, key + ":summary")

    passages = [intro]

    if len(sentences) > 2:
        passages.append(" ".join(sentences[:2]))

    for passage in passages:
        if len(passage) - len(answer) < 40:
            continue

        instruction = template.format(passage=passage)
        example = wiki_example(article, "summary", instruction, answer)

        if fits(tokenizer, example["prompt"], example["response"], 8, 90):
            return example

    return None


def _add_fact(options, prompt, responses, key):
    if not responses:
        return

    response = choose(responses, key)
    words = response.split()

    if 1 <= len(words) <= 30:
        options.append((prompt, response))


def make_factual(article, title, intro, sentences, key, tokenizer):
    first = sentences[0]
    specific = []

    birth = extract_birth(intro)

    if birth is None:
        span = extract_lifespan(first)

        if span is not None:
            birth = _year_fact(span["birth_year"], span["birth_circa"])

    if birth is not None:
        _add_fact(
            specific,
            f"When was {title} born?",
            _birth_responses(title, birth),
            key + ":birth",
        )

    death = extract_died(intro)

    if death is None:
        span = extract_lifespan(first)

        if span is not None:
            death = _year_fact(span["death_year"], span["death_circa"])

    if death is not None and (
        birth is None or death["year"] > birth["year"]
    ):
        _add_fact(
            specific,
            f"When did {title} die?",
            _death_responses(title, death),
            key + ":death",
        )

    origin = extract_origin(intro)

    if origin is not None:
        when = origin["when"]
        prep = _prep_for(when)
        shown = when["display"]

        _add_fact(
            specific,
            f"When was {title} {origin['verb']}?",
            [
                f"{shown}.",
                f"{prep.capitalize()} {shown}.",
                f"{title} was {origin['verb']} {prep} {shown}.",
            ],
            key + ":origin",
        )

    alias = extract_alias(intro, title)

    if alias is not None:
        _add_fact(
            specific,
            f"What else is {title} called?",
            [
                f"{alias}.",
                f"Also known as {alias}.",
            ],
            key + ":alias",
        )

    # A short noun phrase is the fallback. Prefer a date, an alias,
    # or a founded/released year when the lead actually has one.
    pool = specific

    if not pool:
        kind = extract_kind(first)

        if kind is not None:
            short = _as_sentence(kind["phrase"])
            long = f"{title} {kind['verb']} {kind['phrase']}."
            responses = [short]

            if len(long) < 0.85 * len(first):
                responses.append(long)

            _add_fact(
                pool,
                choose(
                    [
                        f"In a few words, what is {title}?",
                        f"Give a short description of {title}.",
                    ],
                    key + ":kind-prompt",
                ),
                responses,
                key + ":kind",
            )

    if not pool:
        return None

    instruction, response = choose(pool, key + ":fact")
    example = wiki_example(article, "factual", instruction, response)

    if not fits(tokenizer, example["prompt"], example["response"], 1, 40):
        return None

    return example


def make_three_facts(article, title, sentences, key, tokenizer):
    if len(sentences) < 3:
        return None

    chosen = sentences[:3]

    if any(len(sentence) < 25 for sentence in chosen):
        return None

    instruction = choose(
        [
            f"Give three facts about {title}.",
            f"List three facts about {title}.",
        ],
        key + ":facts",
    )
    response = "\n".join(f"- {sentence}" for sentence in chosen)
    example = wiki_example(article, "transform", instruction, response)

    if not fits(tokenizer, example["prompt"], example["response"], 12, 320):
        return None

    return example


def make_simplify(article, title, sentences, key, tokenizer):
    sentence = sentences[0]
    simplified = strip_parens(sentence)

    if simplified == sentence:
        return None

    if len(sentence) - len(simplified) < 6:
        return None

    if len(simplified) < 40:
        return None

    if not simplified.endswith((".", "!", "?")):
        return None

    if simplified.count("(") != simplified.count(")"):
        return None

    instruction = choose(
        [
            f"Rewrite this more simply:\n{sentence}",
            f"Say this in simpler words:\n{sentence}",
        ],
        key + ":simple",
    )
    example = wiki_example(article, "transform", instruction, simplified)

    if not fits(tokenizer, example["prompt"], example["response"], 8, 120):
        return None

    return example


def make_transform(article, title, sentences, key, tokenizer):
    options = []

    facts = make_three_facts(article, title, sentences, key, tokenizer)

    if facts is not None:
        options.append(facts)

    simple = make_simplify(article, title, sentences, key, tokenizer)

    if simple is not None:
        options.append(simple)

    if not options:
        return None

    return choose(options, key + ":transform")


def make_misc(article, title, intro, sentences, key, tokenizer):
    templates = [
        "What is this passage about?\n{passage}",
        "What is the subject of the following text?\n{passage}",
        "Which topic does this paragraph describe?\n{passage}",
    ]
    template = choose(templates, key + ":misc")
    passages = [intro, sentences[0]]

    for passage in passages:
        instruction = template.format(passage=passage)
        example = wiki_example(article, "misc", instruction, title)

        if fits(tokenizer, example["prompt"], example["response"], 1, 40):
            return example

    return None


def possible_examples(article, tokenizer):
    title = (article.get("title") or "").strip()
    text = article.get("text") or ""

    if not title or not text.strip() or len(title) > 180:
        return {}

    if "\n" in title:
        return {}

    intro = get_intro(text)

    if intro is None or is_bad_article(title, intro):
        return {}

    sentences = split_sentences(intro)

    if not sentences:
        return {}

    key = str(article.get("id", title))
    made = {
        "overview": make_overview(
            article, title, intro, sentences, key, tokenizer
        ),
        "definition": make_definition(
            article, title, sentences, key, tokenizer
        ),
        "summary": make_summary(
            article, title, intro, sentences, key, tokenizer
        ),
        "factual": make_factual(
            article, title, intro, sentences, key, tokenizer
        ),
        "transform": make_transform(
            article, title, sentences, key, tokenizer
        ),
        "misc": make_misc(
            article, title, intro, sentences, key, tokenizer
        ),
    }

    return {
        task: example
        for task, example in made.items()
        if example is not None
    }


# ---------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------

def quotas(n, fractions=FRACTIONS):
    raw = {
        task: n * share
        for task, share in fractions.items()
    }
    counts = {
        task: int(value)
        for task, value in raw.items()
    }
    leftover = n - sum(counts.values())
    ranked = sorted(
        fractions,
        key=lambda task: raw[task] - counts[task],
        reverse=True,
    )

    for task in ranked[:leftover]:
        counts[task] += 1

    return counts


def cycle_to(items, n, seed):
    if n <= 0 or not items:
        return []

    rng = random.Random(seed)
    out = []

    while len(out) < n:
        batch = [dict(item) for item in items]
        rng.shuffle(batch)
        out.extend(batch)

    return out[:n]


def add_handwritten(bucket, counts, targets, seed):
    """
    Greetings are entirely handwritten. Rewrite / small-talk QA
    fill about half of their buckets; Wikipedia fills the rest.
    """

    plan = {
        "greeting": (GREETINGS, targets["greeting"]),
        "transform": (TRANSFORMS, targets["transform"] // 2),
        "misc": (MISC_HANDWRITTEN, targets["misc"] // 2),
    }

    for task, (pool, n) in plan.items():
        for example in cycle_to(pool, n, seed + article_hash(task)):
            bucket.append(example)
            counts[task] += 1


def take_examples(possible, bucket, counts, targets):
    open_tasks = [
        task
        for task in possible
        if counts[task] < targets[task]
    ]
    open_tasks.sort(
        key=lambda task: counts[task] / targets[task]
    )
    added = 0

    for task in open_tasks[:MAX_PER_ARTICLE]:
        bucket.append(possible[task])
        counts[task] += 1
        added += 1

    return added


def _full(counts, targets):
    return all(
        counts[task] >= targets[task]
        for task in targets
    )


def build_splits(
    articles,
    tokenizer,
    train_targets,
    val_targets,
    max_articles=MAX_ARTICLES,
    seed=SEED,
):
    train = []
    val = []
    train_counts = {task: 0 for task in train_targets}
    val_counts = {task: 0 for task in val_targets}

    add_handwritten(train, train_counts, train_targets, seed)
    add_handwritten(val, val_counts, val_targets, seed + 1)

    scanned = 0

    for article in articles:
        if _full(train_counts, train_targets) and _full(val_counts, val_targets):
            break

        scanned += 1

        if scanned > max_articles:
            break

        possible = possible_examples(article, tokenizer)

        if not possible:
            continue

        article_id = article.get("id", article.get("title", ""))

        if is_validation_article(article_id):
            if not _full(val_counts, val_targets):
                take_examples(possible, val, val_counts, val_targets)
        else:
            if not _full(train_counts, train_targets):
                take_examples(possible, train, train_counts, train_targets)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)

    return {
        "train": train,
        "val": val,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "scanned": scanned,
    }


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


def _count_by(examples, field):
    counts = {}

    for example in examples:
        key = example.get(field) or "none"
        counts[key] = counts.get(key, 0) + 1

    return counts


def _print_split(name, examples, counts, targets):
    print()
    print(f"{name}: {len(examples):,}")

    for task in targets:
        share = len(examples) and counts[task] / len(examples)
        print(
            f"  {task:<12} {counts[task]:6,}  "
            f"target {targets[task]:6,}  "
            f"{share:5.1%}"
        )

    sources = _count_by(examples, "source")
    handwritten = sources.get("handwritten", 0)
    wiki = sources.get("wiki", 0)
    print(f"  handwritten {handwritten:,}   wiki {wiki:,}")


def _print_samples(examples):
    shown = {}

    for example in examples:
        shown.setdefault(example["task"], example)

    print()
    print("Samples")
    print("-------")

    for task in FRACTIONS:
        example = shown.get(task)

        if example is None:
            continue

        prompt = example["prompt"].replace("\n", " / ")
        response = example["response"].replace("\n", " / ")

        if len(prompt) > 180:
            prompt = prompt[:180] + "..."

        if len(response) > 180:
            response = response[:180] + "..."

        print()
        print(f"[{task}] {prompt}")
        print(f"-> {response}")


def _missing(counts, targets):
    return {
        task: targets[task] - counts[task]
        for task in targets
        if counts[task] < targets[task]
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    train_targets = quotas(TRAIN_EXAMPLES)
    val_targets = quotas(VAL_EXAMPLES)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

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

    print(
        f"Building up to {TRAIN_EXAMPLES:,} train "
        f"and {VAL_EXAMPLES:,} val examples"
    )
    print(f"Output: {OUTPUT_DIR}  (data/sft is left unchanged)")

    # Handwritten examples are added inside build_splits before this
    # loop reports anything, so wrap the stream with a bar on articles.
    article_bar = tqdm(
        wiki,
        total=MAX_ARTICLES,
        desc="articles",
    )

    result = build_splits(
        article_bar,
        tokenizer,
        train_targets,
        val_targets,
    )
    article_bar.close()

    train_gap = _missing(result["train_counts"], train_targets)
    val_gap = _missing(result["val_counts"], val_targets)

    _print_split(
        "train",
        result["train"],
        result["train_counts"],
        train_targets,
    )
    _print_split(
        "val",
        result["val"],
        result["val_counts"],
        val_targets,
    )
    print()
    print(f"Articles scanned: {result['scanned']:,}")
    _print_samples(result["train"])

    if train_gap or val_gap:
        print()
        print(f"Train still short: {train_gap or 'nothing'}")
        print(f"Val still short:   {val_gap or 'nothing'}")

        train_got = len(result["train"])

        if train_got < int(TRAIN_EXAMPLES * 0.85):
            raise SystemExit(
                "Corpus is too small. Left the existing jsonl files in place."
            )

        print("Writing anyway. The mix is what we could extract safely.")

    write_jsonl(TRAIN_PATH, result["train"])
    write_jsonl(VAL_PATH, result["val"])

    print()
    print(f"Train file: {TRAIN_PATH}")
    print(f"Val file:   {VAL_PATH}")


if __name__ == "__main__":
    main()
