import os
import json
import time
from getpass import getpass

import pandas as pd
from groq import Groq

# FILE SETTINGS

INPUT_FILE = "evaluation_146_questions_with_good_evidence.csv"
OUTPUT_FILE = "ground_truth_143_questions.csv"

MODEL = "llama-3.1-8b-instant"


# GROQ CLIENT

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    print("Groq key loaded:", bool(api_key))

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")

    return Groq(api_key=api_key)


# PROMPT BUILDER

def build_prompt(question, question_type, evidence_text):
    return f"""
You are creating ground truth for a hallucination evaluation dataset.

Use ONLY the evidence text provided.
Do NOT use outside knowledge.
Do NOT guess.
Do NOT add information that is not present in the evidence.

Question:
{question}

Question type:
{question_type}

Evidence text:
{evidence_text}

Instructions:

If question_type is "factual":
- Extract the direct factual answer from the evidence.
- The answer should be short and specific.
- Do not explain beyond what the evidence supports.
- If the evidence does not answer the question, write "INSUFFICIENT_EVIDENCE".

If question_type is "open_ended":
- Do NOT create one perfect answer.
- Extract multiple evidence-supported themes or concepts.
- Use short phrases separated by semicolons.
- Only include themes that appear in the evidence.
- If the evidence does not contain enough information, write "INSUFFICIENT_EVIDENCE".

Return ONLY valid JSON in this exact format:

{{
  "ground_truth_or_concepts": "...",
  "ground_truth_status": "sufficient or insufficient",
  "notes": "brief explanation of why this was selected"
}}
"""

# JSON PARSER

def parse_json_response(response_text):
    """
    Tries to parse the model response as JSON.
    If the model adds extra text, this tries to extract the JSON part.
    """

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

    return {
        "ground_truth_or_concepts": "PARSING_ERROR",
        "ground_truth_status": "error",
        "notes": response_text
    }


# LLM CALL

def generate_ground_truth(client, question, question_type, evidence_text):
    prompt = build_prompt(question, question_type, evidence_text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    return parse_json_response(content)


# MAIN SCRIPT

def main():
    client = get_groq_client()

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "id",
        "source",
        "cleaned_question",
        "question_type",
        "evidence_text"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create output columns if they do not already exist
    if "ground_truth_or_concepts" not in df.columns:
        df["ground_truth_or_concepts"] = ""

    if "ground_truth_status" not in df.columns:
        df["ground_truth_status"] = ""

    if "ground_truth_notes" not in df.columns:
        df["ground_truth_notes"] = ""

    for index, row in df.iterrows():
        question = str(row["cleaned_question"]).strip()
        question_type = str(row["question_type"]).strip().lower()
        evidence_text = str(row["evidence_text"]).strip()

        existing_value = str(row.get("ground_truth_or_concepts", "")).strip()

        # Skip rows already completed
        if existing_value and existing_value.lower() != "nan":
            continue

        # Handle missing evidence
        if evidence_text == "" or evidence_text.lower() == "nan":
            df.at[index, "ground_truth_or_concepts"] = "INSUFFICIENT_EVIDENCE"
            df.at[index, "ground_truth_status"] = "insufficient"
            df.at[index, "ground_truth_notes"] = "No evidence text was provided."
            df.to_csv(OUTPUT_FILE, index=False)
            continue

        # Handle unsupported question types
        if question_type not in ["factual", "open_ended"]:
            df.at[index, "ground_truth_or_concepts"] = "INSUFFICIENT_EVIDENCE"
            df.at[index, "ground_truth_status"] = "insufficient"
            df.at[index, "ground_truth_notes"] = "Question type was not factual or open_ended."
            df.to_csv(OUTPUT_FILE, index=False)
            continue

        print(f"Processing row {index + 1}/{len(df)} | Type: {question_type}")

        try:
            result = generate_ground_truth(
                client=client,
                question=question,
                question_type=question_type,
                evidence_text=evidence_text
            )

            df.at[index, "ground_truth_or_concepts"] = result.get(
                "ground_truth_or_concepts", ""
            )

            df.at[index, "ground_truth_status"] = result.get(
                "ground_truth_status", ""
            )

            df.at[index, "ground_truth_notes"] = result.get(
                "notes", ""
            )

        except Exception as e:
            df.at[index, "ground_truth_or_concepts"] = "ERROR"
            df.at[index, "ground_truth_status"] = "error"
            df.at[index, "ground_truth_notes"] = str(e)

        # Save after every row so progress is not lost
        df.to_csv(OUTPUT_FILE, index=False)

        # Small delay to reduce rate-limit issues
        time.sleep(1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Saved as {OUTPUT_FILE}")

    print("\nGround truth status counts:")
    print(df["ground_truth_status"].value_counts(dropna=False))

    print("\nQuestion type counts:")
    print(df["question_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
