import pandas as pd

OUTPUT_FILE = "master_raw_questions_all_sources.csv"

cra = pd.read_csv("cra_raw_questions.csv")
osc = pd.read_csv("osc_raw_questions.csv")
reddit = pd.read_csv("reddit_finance_questions.csv")

cra["source"] = "CRA"
osc["source"] = "OSC_GetSmarterAboutMoney"
reddit["source"] = "reddit_finance_questions"

master = pd.concat([cra, osc, reddit], ignore_index=True)

master["question"] = master["question"].astype(str).str.strip()

master = master.drop_duplicates(subset=["question"])

master = master.reset_index(drop=True)
master["question_id"] = ["Q" + str(i + 1).zfill(5) for i in range(len(master))]

master.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print("Saved as:", OUTPUT_FILE)
print("Total rows:", len(master))
print(master["source"].value_counts())