"""
Quality screening only — no factual/open_ended categorization.
For each raw question, Mistral checks if it's usable (clear, relevant,
answerable) and produces a cleaned version of the question text.

Output: final_output.csv with columns:
  question, cleaned_question, usable

Prereqs:
  - ollama serve running
  - mistral pulled
  - pip install pandas ollama tqdm
"""

import pandas as pd
import ollama
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MODEL = "mistral"

BATCH_SIZE = 20        # chunks sent to thread pool
MAX_WORKERS = 4         # parallel calls (increase if strong laptop)
CHECKPOINT_FILE = "checkpoint_v2.csv"


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def screen(question):
    prompt = f"""
You are screening a question for a Canadian retail investment research study.
Check if the question is clear, relevant to personal finance/investing, and answerable.
Also produce a cleaned, well-formed version of the question.

Return ONLY JSON:

{{
  "cleaned_question": "...",
  "usable": true or false
}}

Question:
{question}
"""

    try:
        res = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        data = extract_json(res["message"]["content"])

        if data:
            return data

    except Exception:
        pass

    return {
        "cleaned_question": question,
        "usable": True
    }


def process_batch(batch):
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(screen, q): q for q in batch}

        for future in as_completed(futures):
            q = futures[future]

            try:
                out = future.result()

                results.append({
                    "question": q,
                    "cleaned_question": out.get("cleaned_question", q),
                    "usable": out.get("usable", True)
                })

            except Exception:
                results.append({
                    "question": q,
                    "cleaned_question": q,
                    "usable": False
                })

    return results


def main():
    df = pd.read_csv("master_clean_questions.csv")

    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):

        batch = df.iloc[i:i + BATCH_SIZE]["question"].tolist()

        batch_results = process_batch(batch)

        results.extend(batch_results)

        # checkpoint every batch
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)

    pd.DataFrame(results).to_csv("final_output.csv", index=False)

    print("DONE — SCREENING COMPLETE (no categorization)")


if __name__ == "__main__":
    main()
