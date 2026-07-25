import requests
import pandas as pd
import time

search_terms = [
    "TFSA",
    "RRSP",
    "FHSA",
    "ETF",
    "XEQT",
    "VFV",
    "Wealthsimple",
    "investing",
    "stocks",
    "retirement",
    "risk"
]

questions = []

headers = {
    "User-Agent": "python:mrp.project:v1.0 (by /u/TemporaryUser)"
}

for term in search_terms:

    print(f"Searching for: {term}")

    url = f"https://www.reddit.com/r/PersonalFinanceCanada/search.json?q={term}&restrict_sr=1&limit=100"

    try:

        response = requests.get(url, headers=headers)

        print("Status code:", response.status_code)

        if response.status_code == 200:

            data = response.json()

            posts = data["data"]["children"]

            for post in posts:

                title = post["data"]["title"]

                questions.append({
                    "question": title,
                    "search_term": term,
                    "score": post["data"]["score"],
                    "url": "https://reddit.com" + post["data"]["permalink"]
                })

        else:
            print("Request failed")

    except Exception as e:
        print("Error:", e)

    time.sleep(2)

if len(questions) > 0:

    df = pd.DataFrame(questions)

    df = df.drop_duplicates(subset=["question"])

    df = df[df["question"].str.len() > 15]

    df.to_csv("reddit_finance_questions.csv", index=False)

    print(f"\nCollected {len(df)} questions")

    print(df.head())

else:
    print("\nNo questions collected.")
