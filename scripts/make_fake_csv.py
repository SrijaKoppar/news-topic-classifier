"""Write a small FAKE csv to smoke-test the code paths.

The text is random template filler. Never use it for results or the README.
Run from the repo root:  python -m scripts.make_fake_csv
"""
import csv
import random

from src.config import LABELS, REQUIRED_COLUMNS, ROOT_DIR

KEYWORDS = {
    "POLITICS": ["senate", "election", "governor", "ballot"],
    "WORLD": ["embassy", "border", "treaty", "refugees"],
    "BUSINESS & MONEY": ["earnings", "inflation", "startup", "mortgage"],
    "TECH & SCIENCE": ["algorithm", "telescope", "battery", "genome"],
    "ENTERTAINMENT": ["film", "album", "premiere", "actor"],
    "SPORTS": ["playoffs", "striker", "marathon", "coach"],
    "HEALTH & WELLNESS": ["sleep", "vaccine", "therapy", "nutrition"],
    "LIFESTYLE": ["recipe", "wedding", "travel", "garden"],
    "EDUCATION": ["campus", "tuition", "teachers", "curriculum"],
}
OPENERS = ["New", "Why", "How", "Report:", "Inside the"]
TAILS = ["explained", "surprises experts", "sparks debate", "this week"]

rng = random.Random(0)
rows = []
for i in range(600):
    label = rng.choice(LABELS)
    year = rng.randint(2012, 2022)
    rows.append({
        "id": f"fake{i:04d}",
        "headline": f"{rng.choice(OPENERS)} {rng.choice(KEYWORDS[label])} {rng.choice(TAILS)}",
        "short_description": f"A note on {rng.choice(KEYWORDS[label])} and "
                             f"{rng.choice(KEYWORDS[rng.choice(LABELS)])}.",
        "authors": rng.choice(["", "A. Writer", "B. Author"]),
        "link": f"https://example.invalid/{i}",
        "date": f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "category": label,
        "split_a": "test" if rng.random() < 0.2 else "train",
        "split_b": "test" if year >= 2021 else "train",
    })

# Inject oddities so the report has something to find.
for i in range(10, 40):
    rows[i]["short_description"] = ""                      # blank descriptions
rows[50].update(headline=rows[49]["headline"],             # same text, same label
                short_description=rows[49]["short_description"],
                category=rows[49]["category"])
rows[60].update(headline=rows[59]["headline"],             # same text, different label
                short_description=rows[59]["short_description"],
                category=next(l for l in LABELS if l != rows[59]["category"]))

out = ROOT_DIR / "data" / "fake_sample.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
print("Wrote FAKE data to", out)