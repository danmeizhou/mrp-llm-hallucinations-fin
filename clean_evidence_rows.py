import pandas as pd

INPUT_FILE = "evaluation_150_questions_with_evidence.csv"
OUTPUT_FILE = "evaluation_146_questions_with_good_evidence.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only rows with actual evidence text
clean_df = df[
    df["evidence_text"].notna() &
    (df["evidence_text"].astype(str).str.strip() != "") &
    (~df["evidence_status"].astype(str).str.lower().isin(["error", "insufficient", "discard"]))
].copy()

clean_df.to_csv(OUTPUT_FILE, index=False)

print(f"Original rows: {len(df)}")
print(f"Clean rows with good evidence: {len(clean_df)}")
print(f"Saved as {OUTPUT_FILE}")

print("\nEvidence status counts:")
print(clean_df["evidence_status"].value_counts(dropna=False))

print("\nQuestion type counts:")
print(clean_df["question_type"].value_counts(dropna=False))

print("\nSource counts:")
print(clean_df["source"].value_counts(dropna=False))