import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin


BASE_URL = "https://www.getsmarteraboutmoney.ca"

START_PAGES = [
    "https://www.getsmarteraboutmoney.ca/learning-path/understanding-risk/",
    "https://www.getsmarteraboutmoney.ca/learning-path/etfs/",
    "https://www.getsmarteraboutmoney.ca/learning-path/mutual-funds-segregated-funds/",
    "https://www.getsmarteraboutmoney.ca/learning-path/fraud/",
    "https://www.getsmarteraboutmoney.ca/learning-path/building-your-investing-strategy/",
    "https://www.getsmarteraboutmoney.ca/learning-path/working-with-an-advisor/",
    "https://www.getsmarteraboutmoney.ca/tools/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (MRP academic research; contact: student)"
}


def clean_text(text):
    """Clean extra spaces and line breaks."""
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_links_from_page(url):
    """Extract internal article links from a GetSmarterAboutMoney page."""
    links = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        print("Visited:", url, "Status:", response.status_code)

        if response.status_code != 200:
            return links

        soup = BeautifulSoup(response.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            full_url = urljoin(BASE_URL, href)

            if "getsmarteraboutmoney.ca" in full_url:
                if "/learning-path/" in full_url or "/tools/" in full_url:
                    links.append(full_url.split("#")[0])

    except Exception as e:
        print("Error collecting links:", e)

    return list(set(links))


def scrape_article(url):
    """Scrape article title, headings, and paragraph snippets."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = clean_text(soup.find("h1").get_text()) if soup.find("h1") else ""

        headings = []
        for tag in soup.find_all(["h2", "h3"]):
            text = clean_text(tag.get_text())
            if len(text) > 5:
                headings.append(text)

        paragraphs = []
        for p in soup.find_all("p"):
            text = clean_text(p.get_text())
            if 40 <= len(text) <= 300:
                paragraphs.append(text)

        return {
            "source": "OSC_GetSmarterAboutMoney",
            "title": title,
            "url": url,
            "headings": " | ".join(headings[:10]),
            "snippets": " | ".join(paragraphs[:5])
        }

    except Exception as e:
        print("Error scraping article:", e)
        return None


def categorize_topic(text):
    """Assign an initial MRP category based on keywords."""
    text = text.lower()

    if any(k in text for k in ["risk", "tolerance", "diversification", "loss", "volatile"]):
        return "risk_and_suitability"

    if any(k in text for k in ["etf", "mutual fund", "fund", "fees", "portfolio", "stocks", "bonds"]):
        return "product_recommendations"

    if any(k in text for k in ["fraud", "scam", "complaint", "registration", "advisor"]):
        return "regulatory_compliance"

    if any(k in text for k in ["return", "benchmark", "performance"]):
        return "market_data_claims"

    return "other"


def generate_questions(title, headings):
    """
    Convert OSC article titles/headings into retail-investor style candidate questions.
    This is not final ground truth yet. These are raw candidate questions.
    """
    text_items = []

    if title:
        text_items.append(title)

    if headings:
        text_items.extend(headings.split(" | "))

    questions = []

    for item in text_items:
        item_clean = clean_text(item)

        if len(item_clean) < 8:
            continue

        lower = item_clean.lower()

        if "risk" in lower:
            questions.append(f"What risks should I consider before investing in {item_clean}?")
            questions.append(f"How does {item_clean} affect my investment decisions?")

        elif "etf" in lower:
            questions.append(f"Are ETFs a good investment for a beginner investor?")
            questions.append(f"What should I know before investing in ETFs?")

        elif "mutual fund" in lower or "fund" in lower:
            questions.append(f"What should I check before buying a mutual fund?")
            questions.append(f"Are mutual funds safer than ETFs?")

        elif "fraud" in lower or "scam" in lower:
            questions.append(f"How can I tell if an investment opportunity is a scam?")
            questions.append(f"What should I do before trusting someone offering an investment?")

        elif "advisor" in lower:
            questions.append(f"How do I know if a financial advisor is properly registered?")
            questions.append(f"What questions should I ask before working with an advisor?")

        elif "fee" in lower:
            questions.append(f"How do investment fees affect my returns over time?")
            questions.append(f"Should I choose an investment only because it has low fees?")

        elif "diversification" in lower:
            questions.append(f"Does diversification guarantee that I will not lose money?")
            questions.append(f"How much diversification does a retail investor need?")

        else:
            questions.append(f"What should I know about {item_clean} before investing?")
            questions.append(f"How does {item_clean} affect a retail investor?")

    return list(set(questions))


def main():
    all_links = []

    print("\nCollecting article links...\n")

    for page in START_PAGES:
        links = get_links_from_page(page)
        all_links.extend(links)
        time.sleep(1)

    all_links = list(set(all_links))

    print(f"\nFound {len(all_links)} unique candidate links.\n")

    articles = []

    for link in all_links:
        article = scrape_article(link)

        if article:
            combined_text = article["title"] + " " + article["headings"] + " " + article["snippets"]
            article["category_candidate"] = categorize_topic(combined_text)

            raw_questions = generate_questions(article["title"], article["headings"])

            for q in raw_questions:
                articles.append({
                    "source": article["source"],
                    "question": q,
                    "article_title": article["title"],
                    "category_candidate": article["category_candidate"],
                    "reference_url": article["url"],
                    "raw_headings": article["headings"],
                    "raw_snippets": article["snippets"]
                })

        time.sleep(1)

    df = pd.DataFrame(articles)

    if len(df) > 0:
        df = df.drop_duplicates(subset=["question", "reference_url"])
        df = df[df["question"].str.len() > 25]

        df.to_csv("osc_raw_questions.csv", index=False)

        print("\nOSC raw question dataset created.")
        print("File saved as: osc_raw_questions.csv")
        print("Total questions collected:", len(df))
        print(df.head(10))

    else:
        print("No OSC questions collected.")


if __name__ == "__main__":
    main()