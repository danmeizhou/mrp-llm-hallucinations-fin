"""
Step 1: Generate LLM answers for the 126 questions using multiple Groq models.

Reads : ground_truth_126_sufficient_questions.csv  (must be in the same folder)
Writes: llm_answers_126_<model>.csv  (one file per model)

Usage:
    set GROQ_API_KEY as an environment variable, then:
    python step1_generate_answers_multimodel.py

Notes:
- The question-only prompt matches the original design: the model does NOT
  see the evidence, so answers reflect its own knowledge.
- Progress is saved after every row, so if the script stops (rate limit,
  network), just rerun it and it resumes where it left off.
- Edit SUBJECT_MODELS below. Check current model names at:
  https://console.groq.com/docs/models
"""

import os
import time
import requests
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
INPUT_FILE = "ground_truth_126_sufficient_questions.csv"

SUBJECT_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    # "llama-3.1-8b-instant",  # uncomment only if you want to regenerate
    #                           # the original model's answers as well
]

TEMPERATURE = 0.0
MAX_TOKENS = 512
SLEEP_BETWEEN_CALLS = 2.5   # seconds; raise if you hit rate limits often
MAX_RETRIES = 6
# -----------------------------------------------------------------------------

API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit("Set GROQ_API_KEY first, e.g.  export GROQ_API_KEY=gsk_...")

ANSWER_PROMPT = (
    "You are answering a Canadian personal finance and retail investment "
    "question. Answer the question directly and concisely in 2-5 sentences. "
    "Do not add disclaimers.\n\nQuestion: {question}\n\nAnswer:"
)


def call_groq(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.post(API_URL, json=payload, headers=headers, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 429:
            wait = min(60, 5 * attempt)
            print(f"    rate limited, waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
        if r.status_code in (400, 404):
            raise SystemExit(
                f"Model '{model}' rejected ({r.status_code}): {r.text[:300]}\n"
                "Check the model name at https://console.groq.com/docs/models"
            )
        print(f"    HTTP {r.status_code}, retrying (attempt {attempt})")
        time.sleep(5 * attempt)
    return "ERROR_NO_RESPONSE"


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def main() -> None:
    questions = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")

    for model in SUBJECT_MODELS:
        out_file = f"llm_answers_126_{safe_name(model)}.csv"
        if os.path.exists(out_file):
            done = pd.read_csv(out_file)
            done_ids = set(done["id"])
            rows = done.to_dict("records")
            print(f"[{model}] resuming: {len(done_ids)} answers already saved")
        else:
            done_ids, rows = set(), []
            print(f"[{model}] starting fresh")

        for _, q in questions.iterrows():
            if q["id"] in done_ids:
                continue
            answer = call_groq(model, ANSWER_PROMPT.format(question=q["cleaned_question"]))
            rows.append(
                {
                    "id": q["id"],
                    "source": q["source"],
                    "cleaned_question": q["cleaned_question"],
                    "question_type": q["question_type"],
                    "model": model,
                    "llm_answer": answer,
                }
            )
            pd.DataFrame(rows).to_csv(out_file, index=False)
            print(f"[{model}] {len(rows)}/{len(questions)} id={q['id']}")
            time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"[{model}] done -> {out_file}")

    print("\nAll models finished. Next: run step2_evaluate_answers.py")


if __name__ == "__main__":
    main()
