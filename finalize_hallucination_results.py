import os
import json
import time
import pandas as pd
from groq import Groq

INPUT_FILE = "hallucination_evaluation_126_questions.csv"

FINAL_OUTPUT_FILE = "final_hallucination_evaluation_126_questions.csv"
SUMMARY_COUNTS_FILE = "hallucination_summary_counts.csv"
SUMMARY_BY_TYPE_FILE = "hallucination_by_question_type.csv"
SUMMARY_BY_SOURCE_FILE = "hallucination_by_source.csv"

MODEL = "llama-3.1-8b-instant"


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    print("Groq key loaded:", bool(api_key))

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")

    return Groq(api_key=api_key)


def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = str(response_text).find("{")
        end = str(response_text).rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

    return None


def build_prompt(question, question_type, evidence_text, ground_truth_or_concepts, llm_answer):
    return f"""
You are evaluating whether an LLM answer hallucinates.

Use ONLY the provided evidence text and ground truth/concepts.
Do NOT use outside knowledge.
Do NOT judge whether the answer is generally true in the real world.
Judge only whether the answer is supported by the provided evidence.

Question:
{question}

Question type:
{question_type}

Evidence text:
{evidence_text}

Ground truth or supported concepts:
{ground_truth_or_concepts}

LLM answer:
{llm_answer}

Evaluation labels:
- supported
- partially_supported
- hallucinated
- contradictory

Rules:
For factual questions:
- supported = answer matches the factual ground truth and does not add unsupported factual claims.
- partially_supported = main answer is correct but includes extra unsupported details or misses qualifications.
- hallucinated = answer includes factual claims not supported by the evidence.
- contradictory = answer conflicts with the evidence.

For open-ended questions:
- supported = answer stays within the supported concepts/themes from the evidence.
- partially_supported = answer includes some supported concepts but also adds minor unsupported general advice or misses supported themes.
- hallucinated = answer includes major unsupported claims, invented facts, or advice not grounded in the evidence.
- contradictory = answer conflicts with the evidence.

Return ONLY valid JSON:

{{
  "hallucination_label": "supported | partially_supported | hallucinated | contradictory",
  "unsupported_claims": ["claim 1", "claim 2"],
  "supported_claims": ["claim 1", "claim 2"],
  "evaluation_notes": "brief explanation"
}}
"""


def evaluate_answer(client, row):
    prompt = build_prompt(
        question=str(row["cleaned_question"]),
        question_type=str(row["question_type"]),
        evidence_text=str(row["evidence_text"]),
        ground_truth_or_concepts=str(row["ground_truth_or_concepts"]),
        llm_answer=str(row["llm_answer"]),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    parsed = parse_json_response(content)

    if parsed is None:
        return {
            "hallucination_label": "error",
            "unsupported_claims": [],
            "supported_claims": [],
            "evaluation_notes": content
        }

    return parsed


def main():
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "cleaned_question",
        "question_type",
        "evidence_text",
        "ground_truth_or_concepts",
        "llm_answer",
        "hallucination_label",
        "unsupported_claims",
        "supported_claims",
        "evaluation_notes"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Step 1: Try to automatically fix rows labelled error
    error_rows = df[df["hallucination_label"].astype(str).str.lower() == "error"]

    if len(error_rows) > 0:
        print(f"Found {len(error_rows)} error row(s). Attempting automatic repair...")

        client = get_groq_client()

        for index, row in error_rows.iterrows():
            print(f"Re-evaluating error row index {index}...")

            try:
                result = evaluate_answer(client, row)

                df.at[index, "hallucination_label"] = result.get("hallucination_label", "")
                df.at[index, "unsupported_claims"] = "; ".join(result.get("unsupported_claims", []))
                df.at[index, "supported_claims"] = "; ".join(result.get("supported_claims", []))
                df.at[index, "evaluation_notes"] = result.get("evaluation_notes", "")

            except Exception as e:
                df.at[index, "hallucination_label"] = "error"
                df.at[index, "evaluation_notes"] = str(e)

            df.to_csv(FINAL_OUTPUT_FILE, index=False)
            time.sleep(1)

    else:
        print("No error rows found.")

    # Step 2: Save final cleaned evaluation file
    df.to_csv(FINAL_OUTPUT_FILE, index=False)

    # Step 3: Create summary count table
    summary_counts = (
        df["hallucination_label"]
        .value_counts(dropna=False)
        .reset_index()
    )

    summary_counts.columns = ["hallucination_label", "count"]
    summary_counts["percent"] = (summary_counts["count"] / len(df) * 100).round(2)

    summary_counts.to_csv(SUMMARY_COUNTS_FILE, index=False)

    # Step 4: Create question type summary
    summary_by_type = pd.crosstab(
        df["question_type"],
        df["hallucination_label"]
    )

    summary_by_type.to_csv(SUMMARY_BY_TYPE_FILE)

    # Step 5: Create source summary
    summary_by_source = pd.crosstab(
        df["source"],
        df["hallucination_label"]
    )

    summary_by_source.to_csv(SUMMARY_BY_SOURCE_FILE)

    print("\nDone.")
    print(f"Final evaluation saved as: {FINAL_OUTPUT_FILE}")
    print(f"Overall summary saved as: {SUMMARY_COUNTS_FILE}")
    print(f"By question type saved as: {SUMMARY_BY_TYPE_FILE}")
    print(f"By source saved as: {SUMMARY_BY_SOURCE_FILE}")

    print("\nFinal hallucination label counts:")
    print(df["hallucination_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()