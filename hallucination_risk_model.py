"""
hallucination_risk_model.py — single-model version (Llama-3.1-8B)
Predicts P(hallucination) from question characteristics.
"""

import os, re, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textstat
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("xgboost unavailable — skipping it in the model comparison.")

warnings.filterwarnings("ignore")

#CONFIG
CSV_PATH = "/Users/jessicazhou/Desktop/MRP PROJECT/pythonProject2/final_hallucination_evaluation_126_questions.csv"
JUDGE_MODEL = "llama-3.3-70b-versatile"
CACHE_PATH = "llm_question_scores_cache.csv"
OUT_DIR = "risk_model_outputs"
SEED = 42
FOLDS = 5

REG_KEYWORDS = ["cra","tax","tfsa","rrsp","fhsa","resp","contribution","withholding",
                "deduction","withdrawal","eligib","osc","regulat","securities",
                "compliance","fraud","protection","registered"]

NUMERIC = ["question_length","readability_fk","evidence_length",
           "retrieval_similarity","ambiguity","reasoning_depth"]
BINARY = ["numerical_content","topic_regulation"]

# load
def load_data():
    df = pd.read_csv(CSV_PATH)
    data = pd.DataFrame({
        "question": df["cleaned_question"].astype(str),
        "source": df["source"].astype(str).str.strip().str.upper(),
        "question_type": df["question_type"].astype(str).str.strip().str.lower(),
        "evidence": df["evidence_text"].astype(str),
        "label": df["hallucination_label"].astype(str).str.strip().str.lower().str.replace(" ", "_"),
    })
    valid = {"supported","partially_supported","hallucinated","contradictory"}
    n0 = len(data)
    data = data[data["label"].isin(valid)].reset_index(drop=True)
    print(f"Loaded {n0} rows; {len(data)} with valid labels.")
    print(data["label"].value_counts(), "\n")
    return data

#text features
def add_text_features(data):
    q = data["question"]
    data["question_length"] = q.str.split().str.len()
    data["numerical_content"] = q.str.contains(r"\d|\$|%|\bpercent\b", case=False).astype(int)

    def _syllables(word):
        word = re.sub(r"[^a-z]", "", word.lower())
        if not word:
            return 1
        groups = re.findall(r"[aeiouy]+", word)
        n = len(groups)
        if word.endswith("e") and n > 1 and not word.endswith(("le", "ee")):
            n -= 1
        return max(n, 1)

    def _fk_grade(text):
        words = re.findall(r"[A-Za-z']+", text)
        if len(words) < 3:
            return np.nan
        sentences = max(len(re.findall(r"[.!?]+", text)), 1)
        syll = sum(_syllables(w) for w in words)
        return 0.39 * (len(words) / sentences) + 11.8 * (syll / len(words)) - 15.59

    data["readability_fk"] = q.apply(_fk_grade)
    data["readability_fk"] = data["readability_fk"].fillna(data["readability_fk"].median())


    data["topic_regulation"] = q.str.lower().str.contains("|".join(REG_KEYWORDS)).astype(int)
    data["evidence_length"] = data["evidence"].str.split().str.len()
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf.fit(pd.concat([data["question"], data["evidence"]]).tolist())
    qv, ev = tfidf.transform(data["question"]), tfidf.transform(data["evidence"])
    data["retrieval_similarity"] = [float(cosine_similarity(qv[i], ev[i])[0,0]) for i in range(len(data))]
    return data

#LLM-scored features
PROMPT = """You are scoring a Canadian personal finance question on two dimensions.

Question: {question}

Score each dimension as an integer from 1 to 5:
- ambiguity: 1 = fully precise and unambiguous, 5 = highly ambiguous or underspecified
- reasoning_depth: 1 = simple fact lookup, 5 = requires multi-step reasoning or weighing trade-offs

Respond with ONLY a JSON object in exactly this format:
{{"ambiguity": <int>, "reasoning_depth": <int>}}"""

def add_llm_features(data):
    cache = pd.read_csv(CACHE_PATH) if os.path.exists(CACHE_PATH) else \
            pd.DataFrame(columns=["question","ambiguity","reasoning_depth"])
    todo = [q for q in data["question"].unique() if q not in set(cache["question"])]
    if todo:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("Set GROQ_API_KEY before running.")
        client = Groq(api_key=key)
        print(f"Scoring {len(todo)} questions with {JUDGE_MODEL} (one-time, cached)...")
        rows = []
        for i, q in enumerate(todo, 1):
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role":"user","content":PROMPT.format(question=q)}],
                        temperature=0.0, max_tokens=60)
                    s = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
                    rows.append({"question":q,"ambiguity":int(s["ambiguity"]),
                                 "reasoning_depth":int(s["reasoning_depth"])})
                    break
                except Exception as e:
                    if attempt == 2:
                        rows.append({"question":q,"ambiguity":np.nan,"reasoning_depth":np.nan})
                        print(f"  failed on question {i}: {e}")
                    time.sleep(2)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} scored")
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(CACHE_PATH, index=False)
    data = data.merge(cache, on="question", how="left")
    for c in ["ambiguity","reasoning_depth"]:
        data[c] = data[c].fillna(data[c].median())
    return data

#design matrix
def build_X(data):
    X = data[NUMERIC + BINARY].copy()
    X["type_factual"] = (data["question_type"] == "factual").astype(int)
    X["source_cra"] = (data["source"] == "CRA").astype(int)
    X["source_osc"] = (data["source"] == "OSC").astype(int)
    return X

def make_outcome(data, tag):
    if tag == "primary":
        return data["label"].isin(["hallucinated","contradictory"]).astype(int), pd.Series(True, index=data.index)
    mask = data["label"].isin(["hallucinated","supported"])
    return (data["label"] == "hallucinated").astype(int)[mask], mask

# models
def logistic_primary(X, y, tag):
    Xs = X.copy()
    Xs[NUMERIC] = StandardScaler().fit_transform(Xs[NUMERIC])
    m = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0, maxiter=200)
    tab = pd.DataFrame({"coef":m.params,"odds_ratio":np.exp(m.params),
                        "ci_low":np.exp(m.conf_int()[0]),"ci_high":np.exp(m.conf_int()[1]),
                        "p_value":m.pvalues}).drop(index="const").round(4)
    tab.to_csv(os.path.join(OUT_DIR, f"logistic_odds_ratios_{tag}.csv"))
    print(f"\n=== Logistic regression ({tag}) — odds ratios (numeric predictors standardized) ===")
    print(tab.to_string())
    print(f"McFadden pseudo-R2: {m.prsquared:.3f}")

def compare_models(X, y, tag):
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    models = {"logistic_regression": LogisticRegression(max_iter=2000),
              "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=SEED)}
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          eval_metric="logloss", random_state=SEED)
    rows, fitted = [], {}
    for name, mdl in models.items():
        auc = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
        rows.append({"model":name,"mean_auc":auc.mean(),"sd_auc":auc.std()})
        fitted[name] = mdl.fit(X, y)
    tab = pd.DataFrame(rows).round(3)
    tab.to_csv(os.path.join(OUT_DIR, f"model_auc_comparison_{tag}.csv"), index=False)
    print(f"\n=== {FOLDS}-fold cross-validated AUC ({tag}) ===")
    print(tab.to_string(index=False))
    trees = ["random_forest"] + (["xgboost"] if HAS_XGB else [])
    best = max(trees, key=lambda n: tab.set_index("model").loc[n,"mean_auc"])
    return fitted[best], best

def run_shap(model, name, X, tag):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, X, show=False)
    p = os.path.join(OUT_DIR, f"shap_summary_{name}_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=200); plt.close()
    imp = pd.DataFrame({"feature":X.columns,"mean_abs_shap":np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False).round(4)
    imp.to_csv(os.path.join(OUT_DIR, f"shap_importance_{name}_{tag}.csv"), index=False)
    print(f"\n=== SHAP feature importance ({name}, {tag}) ===")
    print(imp.to_string(index=False))
    print(f"Plot saved: {p}")

# main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    data = load_data()
    data = add_text_features(data)
    data = add_llm_features(data)
    X_full = build_X(data)
    data.join(X_full, rsuffix="_f").to_csv(os.path.join(OUT_DIR, "feature_table.csv"), index=False)
    for tag in ["primary", "strict"]:
        y, mask = make_outcome(data, tag)
        X = X_full[mask.values]
        print(f"\n########## OUTCOME: {tag} (n={len(y)}, events={int(y.sum())}, rate={y.mean():.1%}) ##########")
        logistic_primary(X, y, tag)
        best, name = compare_models(X, y, tag)
        run_shap(best, name, X, tag)
    print(f"\nAll outputs written to '{OUT_DIR}/'. Done.")

if __name__ == "__main__":
    main()