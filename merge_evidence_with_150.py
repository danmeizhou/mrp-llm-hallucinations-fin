import pandas as pd

# Input files
QUESTIONS_FILE = "categorizing_150_questions.csv"
EVIDENCE_FILE = "original_evidence_dataset.csv"

# Output files
OUTPUT_FILE = "evaluation_150_questions_with_evidence.csv"
MISSING_FILE = "missing_evidence_questions.csv"

# Load files
questions_df = pd.read_csv(QUESTIONS_FILE)
evidence_df = pd.read_csv(EVIDENCE_FILE)

# Normalize text so the merge works even if spacing/capitalization differs
questions_df["merge_key"] = questions_df["cleaned_question"].astype(str).str.strip().str.lower()
evidence_df["merge_key"] = evidence_df["cleaned_question"].astype(str).str.strip().str.lower()

# Keep only evidence columns needed for merging
evidence_df = evidence_df[
    [
        "merge_key",
        "source_url",
        "evidence_text"
    ]
]

# Merge evidence into the 150-question file
merged_df = questions_df.merge(
    evidence_df,
    on="merge_key",
    how="left"
)

# Remove helper column
merged_df = merged_df.drop(columns=["merge_key"])

# Add columns needed for evaluation
merged_df["ground_truth_or_concepts"] = ""
merged_df["llm_answer"] = ""
merged_df["hallucination_label"] = ""
merged_df["notes"] = ""

# Save final merged evaluation file
merged_df.to_csv(OUTPUT_FILE, index=False)

# Save rows still missing evidence
missing_df = merged_df[
    merged_df["evidence_text"].isna() |
    (merged_df["evidence_text"].astype(str).str.strip() == "")
]

missing_df.to_csv(MISSING_FILE, index=False)

print(f"Done! Saved as {OUTPUT_FILE}")
print(f"Missing evidence rows saved as {MISSING_FILE}")

print("\nTotal rows:", len(merged_df))
print("Rows missing evidence:", len(missing_df))

print("\nPreview:")
print(merged_df.head())