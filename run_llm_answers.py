import os
import time
import pandas as pd
from groq import Groq

INPUT_FILE = "ground_truth_126_sufficient_questions.csv"
OUTPUT_FILE = "llm_answers_126_questions.csv"

MODEL = "llama-3.1-8b-instant"


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    print("Groq key loaded:", bool(api_key))

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")

    return Groq(api_key=api_key)


def ask_llm(client, question):
    prompt = f"""
Answer the following question clearly and concisely.

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def main():
    client = get_groq_client()

    df = pd.read_csv(INPUT_FILE)

    if "llm_answer" not in df.columns:
        df["llm_answer"] = ""

    for index, row in df.iterrows():
        question = str(row["cleaned_question"]).strip()

        existing_answer = str(row.get("llm_answer", "")).strip()

        if existing_answer and existing_answer.lower() != "nan":
            continue

        print(f"Processing row {index + 1}/{len(df)}")

        try:
            answer = ask_llm(client, question)
            df.at[index, "llm_answer"] = answer

        except Exception as e:
            df.at[index, "llm_answer"] = f"ERROR: {e}"

        df.to_csv(OUTPUT_FILE, index=False)

        time.sleep(1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done! Saved as {OUTPUT_FILE}")


if __name__ == "__main__":
    main()