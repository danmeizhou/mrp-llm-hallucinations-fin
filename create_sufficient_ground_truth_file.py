import pandas as pd

INPUT_FILE = "ground_truth_143_questions.csv"
OUTPUT_FILE = "ground_truth_126_sufficient_questions.csv"

df = pd.read_csv(INPUT_FILE)

clean_df = df[
    df["ground_truth_status"].astype(str).str.lower() == "sufficient"
].copy()

clean_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {OUTPUT_FILE}")
print("Rows:", len(clean_df))

print("\nQuestion type counts:")
print(clean_df["question_type"].value_counts(dropna=False))

print("\nSource counts:")
print(clean_df["source"].value_counts(dropna=False))