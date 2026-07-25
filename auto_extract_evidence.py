import os
import time
import json
import re
from getpass import getpass
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from groq import Groq

# FILE SETTINGS

INPUT_FILE = "categorizing_150_questions.csv"
OUTPUT_FILE = "evaluation_150_questions_with_evidence.csv"

MODEL = "llama-3.1-8b-instant"

# GROQ SETUP

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Set it in Terminal before running the script."
        )

    return Groq(api_key=api_key)


# SEARCH QUERY BUILDER

def build_search_query(question, source):
    question = str(question).strip()
    source = str(source).strip().lower()

    if source == "cra":
        return f"{question} site:canada.ca revenue agency CRA"

    if source == "osc":
        return f"{question} site:osc.ca OR site:getsmarteraboutmoney.ca"

    if source == "reddit":
        return f"{question} site:reddit.com/r/PersonalFinanceCanada"

    return question


# WEB SEARCH

def search_web(query, max_results=5):
    results = []

    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
    except Exception as e:
        print(f"Search error: {e}")

    return results


# FETCH PAGE TEXT

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def fetch_page_text(url, max_chars=12000):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script/style/navigation junk
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = clean_text(text)

        return text[:max_chars]

    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return ""


# EVIDENCE SELECTION PROMPT

def build_evidence_prompt(question, question_type, source, candidates):
    candidate_text = ""

    for i, item in enumerate(candidates, start=1):
        candidate_text += f"""
SOURCE {i}
Title: {item["title"]}
URL: {item["url"]}
Snippet/Page Text:
{item["text"][:4000]}
"""

    return f"""
You are selecting evidence for a hallucination evaluation dataset.

Question:
{question}

Question type:
{question_type}

Expected source category:
{source}

Candidate source texts:
{candidate_text}

Task:
Select the best evidence text that can support answering the question.

Rules:
- Use ONLY the candidate source texts.
- Do NOT answer the question.
- Do NOT add outside knowledge.
- Choose evidence that directly supports the question.
- For factual questions, evidence should contain the specific fact.
- For open-ended questions, evidence may contain several themes, examples, or viewpoints.
- If none of the candidates contain useful evidence, return INSUFFICIENT_EVIDENCE.

Return ONLY valid JSON in this format:

{{
  "source_url": "...",
  "evidence_text": "...",
  "evidence_status": "found or insufficient",
  "notes": "brief explanation"
}}
"""


def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    return {
        "source_url": "",
        "evidence_text": "",
        "evidence_status": "error",
        "notes": text
    }


def select_best_evidence(client, question, question_type, source, candidates):
    prompt = build_evidence_prompt(question, question_type, source, candidates)

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

    required_columns = ["id", "source", "cleaned_question", "question_type"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add evidence columns
    df["source_url"] = ""
    df["evidence_text"] = ""
    df["evidence_status"] = ""
    df["evidence_notes"] = ""

    for index, row in df.iterrows():
        question = str(row["cleaned_question"]).strip()
        source = str(row["source"]).strip()
        question_type = str(row["question_type"]).strip()

        print(f"\nProcessing {index + 1}/{len(df)}")
        print(f"Question: {question}")
        print(f"Source: {source}")

        if question_type == "discard":
            df.at[index, "evidence_status"] = "discard"
            df.at[index, "evidence_notes"] = "Question marked as discard."
            continue

        query = build_search_query(question, source)
        print(f"Search query: {query}")

        search_results = search_web(query, max_results=5)

        if not search_results:
            df.at[index, "evidence_status"] = "insufficient"
            df.at[index, "evidence_notes"] = "No search results found."
            df.to_csv(OUTPUT_FILE, index=False)
            continue

        candidates = []

        for result in search_results:
            url = result["url"]

            page_text = fetch_page_text(url)

            # If page text fails, use search snippet as fallback
            if not page_text:
                page_text = result["snippet"]

            candidates.append({
                "title": result["title"],
                "url": url,
                "text": page_text
            })

            time.sleep(0.5)

        try:
            selected = select_best_evidence(
                client=client,
                question=question,
                question_type=question_type,
                source=source,
                candidates=candidates
            )

            df.at[index, "source_url"] = selected.get("source_url", "")
            df.at[index, "evidence_text"] = selected.get("evidence_text", "")
            df.at[index, "evidence_status"] = selected.get("evidence_status", "")
            df.at[index, "evidence_notes"] = selected.get("notes", "")

        except Exception as e:
            df.at[index, "evidence_status"] = "error"
            df.at[index, "evidence_notes"] = str(e)

        # Save progress after every row
        df.to_csv(OUTPUT_FILE, index=False)

        # Delay to reduce search/API rate issues
        time.sleep(1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone! Saved as {OUTPUT_FILE}")

    print("\nEvidence status counts:")
    print(df["evidence_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
