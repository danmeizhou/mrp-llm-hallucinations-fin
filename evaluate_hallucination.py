import os
import json
import time
import pandas as pd
from groq import Groq


# FILE SETTINGS

INPUT_FILE = "llm_answers_126_questions.csv"
OUTPUT_FILE = "hallucination_evaluation_126_questions.csv"

MODEL = "llama-3.1-8b-instant"

# GROQ CLIENT

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    print("Groq key loaded:", bool(api_key))

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")

    return Groq(api_key=api_key)


# PROMPT BUILDER

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

Evaluation rules:

For factual questions:
- "supported" means the answer matches the factual ground truth and does not add unsupported factual claims.
- "partially_supported" means the main answer is correct but includes extra unsupported details or misses important qualifications.
- "hallucinated" means the answer includes factual claims not supported by the evidence.
- "contradictory" means the answer conflicts with the evidence.

For open-ended questions:
- "supported" means the answer stays within the supported concepts/themes from the evidence.
- "partially_supported" means the answer includes some supported concepts but also adds minor unsupported general advice or misses important supported themes.
- "hallucinated" means the answer includes major unsupported claims, invented facts, or advice not grounded in the evidence.
- "contradictory" means the answer conflicts with the evidence.

Return ONLY valid JSON in this exact format:

{{
  "hallucination_label": "supported | partially_supported | hallucinated | contradictory",
  "unsupported_claims": ["claim 1", "claim 2"],
  "supported_claims": ["claim 1", "claim 2"],
  "evaluation_notes": "brief explanation"
}}
"""

# JSON PARSER

def parse_json_response(response_text):
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
        "hallucination_label": "error",
        "unsupported_claims": [],
        "supported_claims": [],
        "evaluation_notes": response_text
    }


# LLM EVALUATOR

def evaluate_answer(client, question, question_type, evidence_text, ground_truth_or_concepts, llm_answer):
    prompt = build_prompt(
        question=question,
        question_type=question_type,
        evidence_text=evidence_text,
        ground_truth_or_concepts=ground_truth_or_concepts,
        llm_answer=llm_answer
    )

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
        "cleaned_question",
        "question_type",
        "evidence_text",
        "ground_truth_or_concepts",
        "llm_answer"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create output columns if they do not exist
    if "hallucination_label" not in df.columns:
        df["hallucination_label"] = ""

    if "unsupported_claims" not in df.columns:
        df["unsupported_claims"] = ""

    if "supported_claims" not in df.columns:
        df["supported_claims"] = ""

    if "evaluation_notes" not in df.columns:
        df["evaluation_notes"] = ""

    for index, row in df.iterrows():
        question = str(row["cleaned_question"]).strip()
        question_type = str(row["question_type"]).strip().lower()
        evidence_text = str(row["evidence_text"]).strip()
        ground_truth_or_concepts = str(row["ground_truth_or_concepts"]).strip()
        llm_answer = str(row["llm_answer"]).strip()

        existing_label = str(row.get("hallucination_label", "")).strip()

        # Skip rows already evaluated
        if existing_label and existing_label.lower() != "nan":
            continue

        print(f"Evaluating row {index + 1}/{len(df)} | Type: {question_type}")

        try:
            result = evaluate_answer(
                client=client,
                question=question,
                question_type=question_type,
                evidence_text=evidence_text,
                ground_truth_or_concepts=ground_truth_or_concepts,
                llm_answer=llm_answer
            )

            df.at[index, "hallucination_label"] = result.get("hallucination_label", "")
            df.at[index, "unsupported_claims"] = "; ".join(result.get("unsupported_claims", []))
            df.at[index, "supported_claims"] = "; ".join(result.get("supported_claims", []))
            df.at[index, "evaluation_notes"] = result.get("evaluation_notes", "")

        except Exception as e:
            df.at[index, "hallucination_label"] = "error"
            df.at[index, "unsupported_claims"] = ""
            df.at[index, "supported_claims"] = ""
            df.at[index, "evaluation_notes"] = str(e)

        # Save progress after every row
        df.to_csv(OUTPUT_FILE, index=False)

        time.sleep(1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Saved as {OUTPUT_FILE}")

    print("\nHallucination label counts:")
    print(df["hallucination_label"].value_counts(dropna=False))

    print("\nQuestion type vs hallucination label:")
    print(pd.crosstab(df["question_type"], df["hallucination_label"]))

    print("\nSource vs hallucination label:")
    print(pd.crosstab(df["source"], df["hallucination_label"]))


if __name__ == "__main__":
    main()
