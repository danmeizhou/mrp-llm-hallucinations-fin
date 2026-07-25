import pandas as pd

INPUT_FILE = "evaluation_150_questions_with_evidence.csv"

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))

print("\nEvidence status counts:")
print(df["evidence_status"].value_counts(dropna=False))

print("\nQuestion type counts:")
print(df["question_type"].value_counts(dropna=False))

print("\nSource counts:")
print(df["source"].value_counts(dropna=False))

# Missing evidence
missing = df[
    df["evidence_text"].isna() |
    (df["evidence_text"].astype(str).str.strip() == "")
]

missing.to_csv("missing_evidence_rows.csv", index=False)

# Error or insufficient evidence
problem_status = df[
    df["evidence_status"].astype(str).str.lower().isin(
        ["error", "insufficient", "discard"]
    )
]

problem_status.to_csv("problem_evidence_rows.csv", index=False)

# Very short evidence is often weak
short_evidence = df[
    df["evidence_text"].astype(str).str.len() < 80
]

short_evidence.to_csv("short_evidence_rows.csv", index=False)

# Reddit factual rows are often suspicious because Reddit is usually anecdotal/opinion-based
reddit_factual = df[
    (df["source"].astype(str).str.lower() == "reddit") &
    (df["question_type"].astype(str).str.lower() == "factual")
]

reddit_factual.to_csv("reddit_factual_review_rows.csv", index=False)

print("\nSaved review files:")
print("missing_evidence_rows.csv")
print("problem_evidence_rows.csv")
print("short_evidence_rows.csv")
print("reddit_factual_review_rows.csv")