"""
pooled_risk_model.py — three-model pooled hallucination-risk analysis.
Judges the 70B and GPT-OSS answers with the same evaluator/prompt as the MRP,
pools all three models (n=378), then fits logistic regression + RF
and SHAP.
"""

import os, re, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ------------------------------------------------------------------ CONFIG
DIR = "/Users/jessicazhou/Desktop/MRP PROJECT/pythonProject2"
MAIN_CSV = os.path.join(DIR, "final_hallucination_evaluation_126_questions.csv")
ANSWER_FILES = {
    "llama-3.3-70b-versatile": os.path.join(DIR, "llm_answers_126_llama-3.3-70b-versatile.csv"),
    "gpt-oss-20b":             os.path.join(DIR, "llm_answers_126_openai_gpt-oss-20b.csv"),
}
JUDGE_MODEL = "llama-3.1-8b-instant"       # same judge as the MRP
SCORE_MODEL = "llama-3.3-70b-versatile"    # judge for ambiguity / reasoning depth
LABEL_CACHE = os.path.join(DIR, "regenerated_labels_cache.csv")
SCORE_CACHE = os.path.join(DIR, "llm_question_scores_cache.csv")
OUT_DIR = os.path.join(DIR, "pooled_risk_outputs")
SEED, FOLDS = 42, 5

REG_KEYWORDS = ["cra","tax","tfsa","rrsp","fhsa","resp","contribution","withholding",
                "deduction","withdrawal","eligib","osc","regulat","securities",
                "compliance","fraud","protection","registered"]
NUMERIC = ["question_length","readability_fk","evidence_length",
           "retrieval_similarity","ambiguity","reasoning_depth"]
BINARY = ["numerical_content","topic_regulation"]

def groq_client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("Set GROQ_API_KEY before running.")
    return Groq(api_key=key)

# 1. judge the answers
EVAL_PROMPT = """You are evaluating whether an answer to a Canadian personal finance
question is supported by the provided evidence and ground truth.

Question: {question}
Question type: {question_type}

Evidence text:
{evidence}

Ground truth (direct answer for factual questions; supported
concepts/themes for open-ended questions):
{ground_truth}

Answer to evaluate:
{answer}

Classify the answer using EXACTLY one of these labels:
- supported: fully supported by the evidence and ground truth
- partially_supported: contains supported information but also
  missing context, vague claims, or minor unsupported details
- hallucinated: contains major claims not supported by the evidence
- contradictory: directly conflicts with the evidence or ground truth

Respond in exactly this format:
LABEL: <one label>
NOTES: <one or two sentences explaining the decision>"""

VALID = {"supported","partially_supported","hallucinated","contradictory"}

def regenerate_labels(main):
    cache = pd.read_csv(LABEL_CACHE) if os.path.exists(LABEL_CACHE) else \
            pd.DataFrame(columns=["id","model","label"])
    ref = main.set_index("id")[["evidence_text","ground_truth_or_concepts",
                                "cleaned_question","question_type"]]
    rows, client = [], None
    for model_name, path in ANSWER_FILES.items():
        ans = pd.read_csv(path)
        done = set(cache[cache["model"] == model_name]["id"])
        todo = ans[~ans["id"].isin(done)]
        if len(todo) == 0:
            print(f"{model_name}: labels already cached.")
            continue
        if client is None:
            client = groq_client()
        print(f"Judging {len(todo)} answers for {model_name} with {JUDGE_MODEL}...")
        for i, r in enumerate(todo.itertuples(), 1):
            meta = ref.loc[r.id]
            prompt = EVAL_PROMPT.format(
                question=meta["cleaned_question"], question_type=meta["question_type"],
                evidence=str(meta["evidence_text"])[:6000],
                ground_truth=str(meta["ground_truth_or_concepts"])[:3000],
                answer=str(r.llm_answer)[:4000])
            label = "error"
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.0, max_tokens=150)
                    txt = resp.choices[0].message.content
                    m = re.search(r"LABEL:\s*([a-z_ ]+)", txt, re.I)
                    if m:
                        cand = m.group(1).strip().lower().replace(" ", "_")
                        if cand in VALID:
                            label = cand
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  failed on id {r.id}: {e}")
                    time.sleep(2)
            rows.append({"id": r.id, "model": model_name, "label": label})
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(LABEL_CACHE, index=False)
        rows = []
    return pd.read_csv(LABEL_CACHE)

# ------------------------------------------------------- 2. build pooled data
def build_pooled():
    main = pd.read_csv(MAIN_CSV)
    labels = regenerate_labels(main)

    base = main[["id","source","cleaned_question","question_type",
                 "evidence_text","hallucination_label"]].copy()
    frames = [base.assign(model="llama-3.1-8b-instant")
                  .rename(columns={"hallucination_label":"label"})]
    for model_name, path in ANSWER_FILES.items():
        lab = labels[labels["model"] == model_name][["id","label"]]
        f = base.drop(columns=["hallucination_label"]).merge(lab, on="id", how="inner")
        f["model"] = model_name
        frames.append(f)

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns={"cleaned_question":"question","evidence_text":"evidence"})
    data["question"] = data["question"].astype(str)
    data["evidence"] = data["evidence"].astype(str)
    data["source"] = data["source"].astype(str).str.strip().str.upper()
    data["question_type"] = data["question_type"].astype(str).str.strip().str.lower()
    data["label"] = data["label"].astype(str).str.strip().str.lower().str.replace(" ","_")
    n0 = len(data)
    data = data[data["label"].isin(VALID)].reset_index(drop=True)
    print(f"\nPooled {n0} rows; {len(data)} with valid labels.")
    print(pd.crosstab(data["model"], data["label"]), "\n")
    print("Label percentages by model (compare with Table 5.2):")
    print((pd.crosstab(data["model"], data["label"], normalize="index")*100).round(1), "\n")
    return data

# ------------------------------------------------------- 3. features
def add_features(data):
    q = data["question"]
    data["question_length"] = q.str.split().str.len()
    data["numerical_content"] = q.str.contains(r"\d|\$|%|\bpercent\b", case=False).astype(int)

    def syl(w):
        w = re.sub(r"[^a-z]", "", w.lower())
        if not w: return 1
        n = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and n > 1 and not w.endswith(("le","ee")): n -= 1
        return max(n, 1)

    def fk(text):
        words = re.findall(r"[A-Za-z']+", text)
        if len(words) < 3: return np.nan
        sents = max(len(re.findall(r"[.!?]+", text)), 1)
        return 0.39*(len(words)/sents) + 11.8*(sum(syl(w) for w in words)/len(words)) - 15.59

    data["readability_fk"] = q.apply(fk)
    data["readability_fk"] = data["readability_fk"].fillna(data["readability_fk"].median())
    data["topic_regulation"] = q.str.lower().str.contains("|".join(REG_KEYWORDS)).astype(int)
    data["evidence_length"] = data["evidence"].str.split().str.len()

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf.fit(pd.concat([data["question"], data["evidence"]]).tolist())
    qv, ev = tfidf.transform(data["question"]), tfidf.transform(data["evidence"])
    data["retrieval_similarity"] = [float(cosine_similarity(qv[i], ev[i])[0,0]) for i in range(len(data))]
    return data

PROMPT2 = """You are scoring a Canadian personal finance question on two dimensions.

Question: {question}

Score each dimension as an integer from 1 to 5:
- ambiguity: 1 = fully precise and unambiguous, 5 = highly ambiguous or underspecified
- reasoning_depth: 1 = simple fact lookup, 5 = requires multi-step reasoning or weighing trade-offs

Respond with ONLY a JSON object in exactly this format:
{{"ambiguity": <int>, "reasoning_depth": <int>}}"""

def add_llm_features(data):
    cache = pd.read_csv(SCORE_CACHE) if os.path.exists(SCORE_CACHE) else \
            pd.DataFrame(columns=["question","ambiguity","reasoning_depth"])
    todo = [q for q in data["question"].unique() if q not in set(cache["question"])]
    if todo:
        client = groq_client()
        print(f"Scoring {len(todo)} questions with {SCORE_MODEL}...")
        rows = []
        for i, q in enumerate(todo, 1):
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=SCORE_MODEL,
                        messages=[{"role":"user","content":PROMPT2.format(question=q)}],
                        temperature=0.0, max_tokens=60)
                    s = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
                    rows.append({"question":q,"ambiguity":int(s["ambiguity"]),
                                 "reasoning_depth":int(s["reasoning_depth"])})
                    break
                except Exception as e:
                    if attempt == 2:
                        rows.append({"question":q,"ambiguity":np.nan,"reasoning_depth":np.nan})
                    time.sleep(2)
            if i % 25 == 0: print(f"  {i}/{len(todo)}")
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(SCORE_CACHE, index=False)
    data = data.merge(cache, on="question", how="left")
    for c in ["ambiguity","reasoning_depth"]:
        data[c] = data[c].fillna(data[c].median())
    return data

# ------------------------------------------------------- 4. design + models
def build_X(data):
    X = data[NUMERIC + BINARY].copy()
    X["type_factual"] = (data["question_type"] == "factual").astype(int)
    # CRA + OSC merged: OSC alone has too few questions to estimate
    X["source_official"] = data["source"].isin(["CRA","OSC"]).astype(int)
    X["model_llama70b"] = (data["model"] == "llama-3.3-70b-versatile").astype(int)
    X["model_gptoss20b"] = (data["model"] == "gpt-oss-20b").astype(int)
    return X

def make_outcome(data, tag):
    if tag == "primary":
        return data["label"].isin(["hallucinated","contradictory"]).astype(int), pd.Series(True, index=data.index)
    mask = data["label"].isin(["hallucinated","supported"])
    return (data["label"] == "hallucinated").astype(int)[mask], mask

def logistic(X, y, tag):
    Xs = X.copy(); Xs[NUMERIC] = StandardScaler().fit_transform(Xs[NUMERIC])
    m = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0, maxiter=200)
    tab = pd.DataFrame({"coef":m.params,"odds_ratio":np.exp(m.params),
                        "ci_low":np.exp(m.conf_int()[0]),"ci_high":np.exp(m.conf_int()[1]),
                        "p_value":m.pvalues}).drop(index="const").round(4)
    tab.to_csv(os.path.join(OUT_DIR, f"logistic_odds_ratios_{tag}.csv"))
    print(f"\n=== Logistic regression ({tag}) — odds ratios (numeric standardized) ===")
    print(tab.to_string())
    print(f"McFadden pseudo-R2: {m.prsquared:.3f}")

def compare(X, y, tag):
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
    if isinstance(sv, list): sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3: sv = sv[:, :, 1]
    plt.figure(); shap.summary_plot(sv, X, show=False)
    p = os.path.join(OUT_DIR, f"shap_summary_{name}_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=200); plt.close()
    imp = pd.DataFrame({"feature":X.columns,"mean_abs_shap":np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False).round(4)
    imp.to_csv(os.path.join(OUT_DIR, f"shap_importance_{name}_{tag}.csv"), index=False)
    print(f"\n=== SHAP feature importance ({name}, {tag}) ===")
    print(imp.to_string(index=False))
    print(f"Plot saved: {p}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    data = build_pooled()
    data = add_features(data)
    data = add_llm_features(data)
    X_full = build_X(data)
    data.join(X_full, rsuffix="_f").to_csv(os.path.join(OUT_DIR, "pooled_feature_table.csv"), index=False)
    for tag in ["primary", "strict"]:
        y, mask = make_outcome(data, tag)
        X = X_full[mask.values]
        print(f"\n########## OUTCOME: {tag} (n={len(y)}, events={int(y.sum())}, rate={y.mean():.1%}) ##########")
        logistic(X, y, tag)
        best, name = compare(X, y, tag)
        run_shap(best, name, X, tag)
    print(f"\nAll outputs written to '{OUT_DIR}/'. Done.")

if __name__ == "__main__":
    main()