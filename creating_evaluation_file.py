import pandas as pd

# Load categorized 150-question file
df = pd.read_csv("categorizing_150_questions.csv")

# Remove discarded questions
df = df[df["question_type"] != "discard"].copy()

# Add evaluation columns
df["evidence_text"] = ""
df["ground_truth_or_concepts"] = ""
df["llm_answer"] = ""
df["hallucination_label"] = ""
df["notes"] = ""

# Save evaluation file
df.to_csv("evaluation_150_questions.csv", index=False)

print("Done! Saved as evaluation_150_questions.csv")
print(df.head())

print("\nQuestion type counts:")
print(df["question_type"].value_counts())

print("\nSource counts:")
print(df["source"].value_counts())