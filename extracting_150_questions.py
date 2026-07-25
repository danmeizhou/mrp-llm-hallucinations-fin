import pandas as pd

# Load your screened dataset
df = pd.read_csv("final_output.csv")

# Keep only usable questions
df = df[df["usable"] == True].copy()

# Remove rows where cleaned_question is empty
df = df.dropna(subset=["cleaned_question"])

# Remove duplicate questions
df = df.drop_duplicates(subset=["cleaned_question"])

# If source column does not exist, create it blank for now
if "source" not in df.columns:
    df["source"] = ""

# Randomly extract 150 questions
sample_150 = df.sample(n=150, random_state=42).copy()

# Add ID column
sample_150.insert(0, "id", range(1, len(sample_150) + 1))

# Add question_type column for you to fill manually later
sample_150["question_type"] = ""

# Keep only the columns needed for Phase 1
sample_150 = sample_150[
    [
        "id",
        "source",
        "cleaned_question",
        "question_type"
    ]
]

# Save the extracted questions
sample_150.to_csv("150_questions_output.csv", index=False)

print("Done! Saved as 150_questions_output.csv")
print(sample_150.head())