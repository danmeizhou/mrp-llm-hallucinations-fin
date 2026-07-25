import pandas as pd

INPUT_FILE = "ground_truth_143_questions.csv"

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))

print("\nGround truth status counts:")
print(df["ground_truth_status"].value_counts(dropna=False))

# Save insufficient rows
insufficient = df[df["ground_truth_status"].astype(str).str.lower() == "insufficient"]
insufficient.to_csv("insufficient_ground_truth_rows.csv", index=False)

# Save factual rows for review
factual = df[df["question_type"].astype(str).str.lower() == "factual"]
factual.to_csv("factual_ground_truth_review.csv", index=False)

# Save open-ended rows for review
open_ended = df[df["question_type"].astype(str).str.lower() == "open_ended"]
open_ended.to_csv("open_ended_ground_truth_review.csv", index=False)

print("\nSaved:")
print("insufficient_ground_truth_rows.csv")
print("factual_ground_truth_review.csv")
print("open_ended_ground_truth_review.csv")