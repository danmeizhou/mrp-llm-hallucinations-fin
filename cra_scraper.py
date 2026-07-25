import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time


CRA_PAGES = [
    {
        "topic": "TFSA withdrawals",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/tax-free-savings-account/withdraw.html",
        "category": "tax_advantaged_accounts"
    },
    {
        "topic": "TFSA contributions",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/tax-free-savings-account/contributing.html",
        "category": "tax_advantaged_accounts"
    },
    {
        "topic": "RRSP withdrawals",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans/making-withdrawals.html",
        "category": "tax_advantaged_accounts"
    },
    {
        "topic": "RRSP withdrawal tax rates",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/rrsps-related-plans/making-withdrawals/tax-rates-on-withdrawals.html",
        "category": "tax_advantaged_accounts"
    },
    {
        "topic": "FHSA overview",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/first-home-savings-account.html",
        "category": "tax_advantaged_accounts"
    },
    {
        "topic": "FHSA participation room",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/first-home-savings-account/contributing-your-fhsa.html",
        "category": "tax_advantaged_accounts"
    },
    {
        "topic": "FHSA tax deductions",
        "url": "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/first-home-savings-account/tax-deductions-fhsa-contributions.html",
        "category": "tax_advantaged_accounts"
    }
]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (academic research project)"
}


def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def scrape_cra_page(page):
    try:
        response = requests.get(page["url"], headers=HEADERS, timeout=15)

        print("Scraping:", page["topic"], "| Status:", response.status_code)

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        title = clean_text(soup.find("h1").get_text()) if soup.find("h1") else page["topic"]

        text_blocks = []

        for tag in soup.find_all(["h2", "h3", "p", "li"]):
            text = clean_text(tag.get_text())

            if 40 <= len(text) <= 350:
                text_blocks.append(text)

        rows = []

        for text in text_blocks:
            rows.append({
                "source": "CRA",
                "topic": page["topic"],
                "page_title": title,
                "raw_text": text,
                "reference_url": page["url"],
                "category_candidate": page["category"]
            })

        return rows

    except Exception as e:
        print("Error:", e)
        return []


def generate_question_from_text(text, topic):
    text_lower = text.lower()

    questions = []

    if "tfsa" in text_lower and "withdraw" in text_lower:
        questions.append("Can I recontribute TFSA withdrawals in the same year?")
        questions.append("When does a TFSA withdrawal get added back to my contribution room?")

    if "contribution room" in text_lower:
        questions.append(f"How is contribution room calculated for {topic}?")
        questions.append(f"What happens if I contribute more than my available room for {topic}?")

    if "over-contribution" in text_lower or "excess" in text_lower:
        questions.append(f"What happens if I over-contribute to my {topic}?")

    if "rrsp" in text_lower and "withdraw" in text_lower:
        questions.append("Can I withdraw money from my RRSP before retirement?")
        questions.append("Do I have to pay tax when I withdraw from my RRSP?")

    if "withholds" in text_lower or "withholding" in text_lower:
        questions.append("How much tax is withheld when I withdraw from my RRSP?")

    if "fhsa" in text_lower and "8,000" in text_lower:
        questions.append("What is the annual FHSA contribution limit?")
        questions.append("How much FHSA room do I get when I open my first FHSA?")

    if "40,000" in text_lower:
        questions.append("What is the lifetime FHSA contribution limit?")

    if "deductible" in text_lower:
        questions.append("Are FHSA contributions tax deductible?")
        questions.append("Are transfers from an RRSP to an FHSA tax deductible?")

    if len(questions) == 0:
        questions.append(f"What should I know about {topic}?")

    return list(set(questions))


def main():
    raw_rows = []

    for page in CRA_PAGES:
        page_rows = scrape_cra_page(page)
        raw_rows.extend(page_rows)
        time.sleep(1)

    raw_df = pd.DataFrame(raw_rows)

    if len(raw_df) == 0:
        print("No CRA data collected.")
        return

    raw_df.to_csv("cra_raw_text.csv", index=False)

    question_rows = []

    for _, row in raw_df.iterrows():
        questions = generate_question_from_text(row["raw_text"], row["topic"])

        for q in questions:
            question_rows.append({
                "source": "CRA",
                "question": q,
                "topic": row["topic"],
                "category_candidate": row["category_candidate"],
                "reference_url": row["reference_url"],
                "supporting_text": row["raw_text"]
            })

    questions_df = pd.DataFrame(question_rows)

    questions_df = questions_df.drop_duplicates(subset=["question", "reference_url"])
    questions_df = questions_df[questions_df["question"].str.len() > 20]

    questions_df.to_csv("cra_raw_questions.csv", index=False)

    print("\nCRA raw text saved as: cra_raw_text.csv")
    print("CRA question dataset saved as: cra_raw_questions.csv")
    print("Total raw text rows:", len(raw_df))
    print("Total candidate questions:", len(questions_df))

    print("\nSample questions:")
    print(questions_df.head(15))


if __name__ == "__main__":
    main()