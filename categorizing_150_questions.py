import csv
import re
from pathlib import Path

INPUT_FILE = "150_questions_output.csv"
OUTPUT_FILE = "categorizing_150_questions.csv"


def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def classify_question_type(question):
    q = normalize_text(question)

    discard_patterns = [
        "has anyone",
        "does anyone",
        "anyone else",
        "funny",
        "story",
        "rant",
        "am i cooked",
        "is this normal",
        "thoughts?",
    ]

    open_ended_patterns = [
        "should i",
        "what should",
        "how should",
        "is it worth",
        "is it advisable",
        "would it be better",
        "which is better",
        "best way",
        "recommend",
        "advice",
        "opinion",
        "retire",
        "buy a condo",
        "continue investing",
        "portfolio",
        "investment strategy",
        "financial plan",
        "can i retire",
        "what do users",
        "what do people",
        "why do people",
    ]

    factual_patterns = [
        "what is",
        "what are",
        "when",
        "how much",
        "how many",
        "what happens",
        "do i have to pay tax",
        "is taxable",
        "contribution room",
        "withdrawal",
        "deadline",
        "limit",
        "penalty",
        "withholding tax",
        "eligible",
        "eligibility",
        "requirement",
        "rule",
        "cra",
        "rrsp",
        "tfsa",
        "fhsa",
        "resp",
        "capital gains",
        "tax return",
        "notice of assessment",
    ]

    for pattern in discard_patterns:
        if pattern in q:
            return "discard"

    for pattern in open_ended_patterns:
        if pattern in q:
            return "open_ended"

    for pattern in factual_patterns:
        if pattern in q:
            return "factual"

    # Default rule:
    # Questions with "should", "better", or "recommend" are usually open-ended.
    if re.search(r"\b(should|better|recommend|advice|worth)\b", q):
        return "open_ended"

    # Questions asking what/when/how much are usually factual.
    if re.search(r"^(what|when|how much|how many|where|who)\b", q):
        return "factual"

    return "open_ended"


def classify_source(question):
    q = normalize_text(question)

    cra_keywords = [
        "cra",
        "tax",
        "tfsa",
        "rrsp",
        "fhsa",
        "resp",
        "rrif",
        "cpp",
        "oas",
        "ei",
        "gst",
        "hst",
        "refund",
        "tax return",
        "notice of assessment",
        "contribution room",
        "capital gains",
        "withholding tax",
        "deduction",
        "taxable income",
        "t4",
        "t5",
    ]

    osc_keywords = [
        "osc",
        "securities",
        "investment dealer",
        "advisor",
        "adviser",
        "registrant",
        "prospectus",
        "disclosure",
        "enforcement",
        "exempt market",
        "mutual fund",
        "risk disclosure",
        "suitability",
        "know your client",
        "kyc",
        "complaint",
        "fraud",
        "scam",
        "investor alert",
    ]

    reddit_keywords = [
        "should i",
        "what should i",
        "can i retire",
        "is it worth",
        "advice",
        "opinion",
        "portfolio",
        "wealthsimple",
        "xeqt",
        "veqt",
        "vfv",
        "xgro",
        "vgro",
        "condo",
        "mortgage",
        "salary",
        "my friend",
        "personal",
        "anyone",
    ]

    cra_score = sum(1 for word in cra_keywords if word in q)
    osc_score = sum(1 for word in osc_keywords if word in q)
    reddit_score = sum(1 for word in reddit_keywords if word in q)

    scores = {
        "CRA": cra_score,
        "OSC": osc_score,
        "Reddit": reddit_score,
    }

    best_source = max(scores, key=scores.get)

    # If no keywords match, default to Reddit because broad personal-finance questions
    # are usually community/advice-style questions.
    if scores[best_source] == 0:
        return "Reddit"

    return best_source


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Make sure it is in the same folder as this Python script."
        )

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    output_rows = []

    for row in rows:
        question = row.get("cleaned_question", "").strip()

        output_rows.append({
            "id": row.get("id", "").strip(),
            "source": classify_source(question),
            "cleaned_question": question,
            "question_type": classify_question_type(question),
        })

    fieldnames = ["id", "source", "cleaned_question", "question_type"]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Done! Saved as {OUTPUT_FILE}")

    # Print summary counts
    source_counts = {}
    type_counts = {}

    for row in output_rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
        type_counts[row["question_type"]] = type_counts.get(row["question_type"], 0) + 1

    print("\nSource counts:")
    for key, value in source_counts.items():
        print(f"{key}: {value}")

    print("\nQuestion type counts:")
    for key, value in type_counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()