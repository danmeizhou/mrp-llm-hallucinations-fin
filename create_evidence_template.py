import pandas as pd

# Input: your categorized 150 questions
INPUT_FILE = "categorizing_150_questions.csv"

# Output: file where evidence will be filled
OUTPUT_FILE = "original_evidence_dataset.csv"

# Load categorized questions
df = pd.read_csv(INPUT_FILE)

# Keep only the columns needed for evidence collection
evidence_df = df[["cleaned_question", "source"]].copy()

# Add evidence columns
evidence_df["source_url"] = ""
evidence_df["evidence_text"] = ""

# Remove duplicates if any
evidence_df = evidence_df.drop_duplicates(subset=["cleaned_question"])

# Save evidence template
evidence_df.to_csv(OUTPUT_FILE, index=False)

print(f"Done! Saved as {OUTPUT_FILE}")
print(evidence_df.head())