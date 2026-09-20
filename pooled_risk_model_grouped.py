import os
import re
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap

warnings.filterwarnings("ignore")

#CONFIG

DIR = "/Users/jessicazhou/Desktop/MRP PROJECT/pythonProject2"

MAIN_CSV = os.path.join(
    DIR, "final_hallucination_evaluation_126_questions.csv"
)

ANSWER_FILES = {
    "llama-3.3-70b-versatile": os.path.join(
        DIR, "llm_answers_126_llama-3.3-70b-versatile.csv"
    ),
    "gpt-oss-20b": os.path.join(
        DIR, "llm_answers_126_openai_gpt-oss-20b.csv"
    ),
}

JUDGE_MODEL = "llama-3.1-8b-instant"
SCORE_MODEL = "llama-3.3-70b-versatile"

LABEL_CACHE = os.path.join(DIR, "regenerated_labels_cache.csv")
SCORE_CACHE = os.path.join(DIR, "llm_question_scores_cache.csv")

OUT_DIR = os.path.join(DIR, "pooled_risk_outputs_grouped")

SEED = 42
FOLDS = 5

REG_KEYWORDS = [
    "cra", "tax", "tfsa", "rrsp", "fhsa", "resp", "contribution",
    "withholding", "deduction", "withdrawal", "eligib", "osc", "regulat",
    "securities", "compliance", "fraud", "protection", "registered"
]

NUMERIC = [
    "question_length",
    "readability_fk",
    "evidence_length",
    "retrieval_similarity",
    "ambiguity",
    "reasoning_depth",
]

BINARY = [
    "numerical_content",
    "topic_regulation",
]

VALID = {
    "supported",
    "partially_supported",
    "hallucinated",
    "contradictory",
}

#API fallback only

def groq_client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is required only if a cache file is incomplete."
        )
    return Groq(api_key=key)


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


def regenerate_labels(main):
    cache = (
        pd.read_csv(LABEL_CACHE)
        if os.path.exists(LABEL_CACHE)
        else pd.DataFrame(columns=["id", "model", "label"])
    )

    ref = main.set_index("id")[
        [
            "evidence_text",
            "ground_truth_or_concepts",
            "cleaned_question",
            "question_type",
        ]
    ]

    rows = []
    client = None

    for model_name, path in ANSWER_FILES.items():
        ans = pd.read_csv(path)
        done = set(cache.loc[cache["model"] == model_name, "id"])
        todo = ans[~ans["id"].isin(done)]

        if len(todo) == 0:
            print(f"{model_name}: labels already cached.")
            continue

        if client is None:
            client = groq_client()

        print(f"Judging {len(todo)} answers for {model_name}...")

        for i, r in enumerate(todo.itertuples(), 1):
            meta = ref.loc[r.id]

            prompt = EVAL_PROMPT.format(
                question=meta["cleaned_question"],
                question_type=meta["question_type"],
                evidence=str(meta["evidence_text"])[:6000],
                ground_truth=str(meta["ground_truth_or_concepts"])[:3000],
                answer=str(r.llm_answer)[:4000],
            )

            label = "error"

            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=150,
                    )

                    txt = resp.choices[0].message.content
                    m = re.search(r"LABEL:\s*([a-z_ ]+)", txt, re.I)

                    if m:
                        candidate = (
                            m.group(1)
                            .strip()
                            .lower()
                            .replace(" ", "_")
                        )
                        if candidate in VALID:
                            label = candidate
                    break

                except Exception as exc:
                    if attempt == 2:
                        print(f"Failed on id {r.id}: {exc}")
                    time.sleep(2)

            rows.append(
                {"id": r.id, "model": model_name, "label": label}
            )

            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

        cache = pd.concat(
            [cache, pd.DataFrame(rows)],
            ignore_index=True,
        )
        cache.to_csv(LABEL_CACHE, index=False)
        rows = []

    return pd.read_csv(LABEL_CACHE)


#POOLED DATA

def build_pooled():
    main = pd.read_csv(MAIN_CSV)
    labels = regenerate_labels(main)

    base = main[
        [
            "id",
            "source",
            "cleaned_question",
            "question_type",
            "evidence_text",
            "hallucination_label",
        ]
    ].copy()

    frames = [
        base.assign(model="llama-3.1-8b-instant")
        .rename(columns={"hallucination_label": "label"})
    ]

    for model_name, path in ANSWER_FILES.items():
        lab = labels.loc[
            labels["model"] == model_name,
            ["id", "label"],
        ]

        f = (
            base.drop(columns=["hallucination_label"])
            .merge(lab, on="id", how="inner")
        )
        f["model"] = model_name
        frames.append(f)

    data = pd.concat(frames, ignore_index=True)

    data = data.rename(
        columns={
            "cleaned_question": "question",
            "evidence_text": "evidence",
        }
    )

    data["question"] = data["question"].astype(str)
    data["evidence"] = data["evidence"].astype(str)
    data["source"] = (
        data["source"].astype(str).str.strip().str.upper()
    )
    data["question_type"] = (
        data["question_type"].astype(str).str.strip().str.lower()
    )
    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    n_before = len(data)
    data = data[data["label"].isin(VALID)].reset_index(drop=True)

    print(
        f"\nPooled {n_before} rows; "
        f"{len(data)} rows have valid labels."
    )

    print("\nLabel counts by model:")
    print(pd.crosstab(data["model"], data["label"]))

    print("\nLabel percentages by model:")
    print(
        (
            pd.crosstab(
                data["model"],
                data["label"],
                normalize="index",
            )
            * 100
        ).round(1)
    )

    return data


#FEATURES

def add_features(data):
    q = data["question"]

    data["question_length"] = q.str.split().str.len()

    data["numerical_content"] = q.str.contains(
        r"\d|\$|%|\bpercent\b",
        case=False,
    ).astype(int)

    def syllables(word):
        word = re.sub(r"[^a-z]", "", word.lower())

        if not word:
            return 1

        n = len(re.findall(r"[aeiouy]+", word))

        if (
            word.endswith("e")
            and n > 1
            and not word.endswith(("le", "ee"))
        ):
            n -= 1

        return max(n, 1)

    def flesch_kincaid(text):
        words = re.findall(r"[A-Za-z']+", text)

        if len(words) < 3:
            return np.nan

        sentences = max(
            len(re.findall(r"[.!?]+", text)),
            1,
        )

        return (
            0.39 * (len(words) / sentences)
            + 11.8
            * (
                sum(syllables(w) for w in words)
                / len(words)
            )
            - 15.59
        )

    data["readability_fk"] = q.apply(flesch_kincaid)
    data["readability_fk"] = data["readability_fk"].fillna(
        data["readability_fk"].median()
    )

    data["topic_regulation"] = (
        q.str.lower()
        .str.contains("|".join(REG_KEYWORDS))
        .astype(int)
    )

    data["evidence_length"] = (
        data["evidence"].str.split().str.len()
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )

    tfidf.fit(
        pd.concat(
            [data["question"], data["evidence"]]
        ).tolist()
    )

    q_vectors = tfidf.transform(data["question"])
    e_vectors = tfidf.transform(data["evidence"])

    data["retrieval_similarity"] = [
        float(
            cosine_similarity(
                q_vectors[i],
                e_vectors[i],
            )[0, 0]
        )
        for i in range(len(data))
    ]

    return data


SCORE_PROMPT = """You are scoring a Canadian personal finance question on two dimensions.

Question: {question}

Score each dimension as an integer from 1 to 5:
- ambiguity: 1 = fully precise and unambiguous, 5 = highly ambiguous or underspecified
- reasoning_depth: 1 = simple fact lookup, 5 = requires multi-step reasoning or weighing trade-offs

Respond with ONLY a JSON object in exactly this format:
{{"ambiguity": <int>, "reasoning_depth": <int>}}"""


def add_llm_features(data):
    cache = (
        pd.read_csv(SCORE_CACHE)
        if os.path.exists(SCORE_CACHE)
        else pd.DataFrame(
            columns=[
                "question",
                "ambiguity",
                "reasoning_depth",
            ]
        )
    )

    completed = set(cache["question"])

    todo = [
        q
        for q in data["question"].unique()
        if q not in completed
    ]

    if todo:
        client = groq_client()
        print(
            f"Scoring {len(todo)} uncached questions "
            f"with {SCORE_MODEL}..."
        )

        rows = []

        for i, question in enumerate(todo, 1):
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=SCORE_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": SCORE_PROMPT.format(
                                    question=question
                                ),
                            }
                        ],
                        temperature=0.0,
                        max_tokens=60,
                    )

                    result = json.loads(
                        re.search(
                            r"\{.*\}",
                            response.choices[0].message.content,
                            re.S,
                        ).group(0)
                    )

                    rows.append(
                        {
                            "question": question,
                            "ambiguity": int(
                                result["ambiguity"]
                            ),
                            "reasoning_depth": int(
                                result["reasoning_depth"]
                            ),
                        }
                    )
                    break

                except Exception:
                    if attempt == 2:
                        rows.append(
                            {
                                "question": question,
                                "ambiguity": np.nan,
                                "reasoning_depth": np.nan,
                            }
                        )
                    time.sleep(2)

            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

        cache = pd.concat(
            [cache, pd.DataFrame(rows)],
            ignore_index=True,
        )

        cache.to_csv(SCORE_CACHE, index=False)

    data = data.merge(
        cache,
        on="question",
        how="left",
    )

    for column in ["ambiguity", "reasoning_depth"]:
        data[column] = data[column].fillna(
            data[column].median()
        )

    return data


#DESIGN MATRIX

def build_X(data):
    X = data[NUMERIC + BINARY].copy()

    X["type_factual"] = (
        data["question_type"] == "factual"
    ).astype(int)

    X["source_official"] = (
        data["source"].isin(["CRA", "OSC"])
    ).astype(int)

    X["model_llama70b"] = (
        data["model"] == "llama-3.3-70b-versatile"
    ).astype(int)

    X["model_gptoss20b"] = (
        data["model"] == "gpt-oss-20b"
    ).astype(int)

    return X


def make_outcome(data, tag):
    if tag == "primary":
        mask = pd.Series(True, index=data.index)

        y = data["label"].isin(
            ["hallucinated", "contradictory"]
        ).astype(int)

    elif tag == "strict":
        mask = data["label"].isin(
            ["hallucinated", "supported"]
        )

        y = (
            data.loc[mask, "label"]
            == "hallucinated"
        ).astype(int)

    else:
        raise ValueError(
            "tag must be 'primary' or 'strict'"
        )

    return y, mask


#LOGISTIC INFERENCE MODEL

def logistic_inference(X, y, tag):
    """
    Fits the full-sample statsmodels logistic regression used for
    odds ratios, confidence intervals, p-values, and McFadden pseudo-R2.
    """
    Xs = X.copy()

    scaler = StandardScaler()
    Xs[NUMERIC] = scaler.fit_transform(Xs[NUMERIC])

    model = sm.Logit(
        y,
        sm.add_constant(Xs),
    ).fit(
        disp=0,
        maxiter=200,
    )

    table = pd.DataFrame(
        {
            "coef": model.params,
            "odds_ratio": np.exp(model.params),
            "ci_low": np.exp(
                model.conf_int()[0]
            ),
            "ci_high": np.exp(
                model.conf_int()[1]
            ),
            "p_value": model.pvalues,
        }
    ).drop(index="const")

    table = table.round(4)

    table.to_csv(
        os.path.join(
            OUT_DIR,
            f"logistic_odds_ratios_{tag}.csv",
        )
    )

    print(
        f"\n=== Logistic regression ({tag}) "
        f"odds ratios ==="
    )
    print(table.to_string())

    print(
        f"McFadden pseudo-R2: "
        f"{model.prsquared:.3f}"
    )

    return model


#GROUPED MODEL COMPARISON

def build_models(X):
    """
    Logistic regression standardizes continuous variables INSIDE each
    training fold. RF and Gradient Boosting use the raw predictor values.
    """

    non_numeric = [
        c for c in X.columns if c not in NUMERIC
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC,
            ),
            (
                "other",
                "passthrough",
                non_numeric,
            ),
        ],
        remainder="drop",
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=SEED,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=SEED,
        ),
    }


def grouped_model_comparison(X, y, groups, tag):
    cv = StratifiedGroupKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    models = build_models(X)

    rows = []
    fold_rows = []

    for model_name, estimator in models.items():
        metrics = {
            "auc": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups),
            start=1,
        ):
            model = clone(estimator)

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.fit(X_train, y_train)

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            predictions = (
                probabilities >= 0.50
            ).astype(int)

            fold_metrics = {
                "auc": roc_auc_score(
                    y_test,
                    probabilities,
                ),
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }

            for metric_name, value in fold_metrics.items():
                metrics[metric_name].append(
                    value
                )

            fold_rows.append(
                {
                    "outcome": tag,
                    "model": model_name,
                    "fold": fold,
                    "n_test": len(test_idx),
                    "events_test": int(
                        y_test.sum()
                    ),
                    **fold_metrics,
                }
            )

        row = {
            "model": model_name,
        }

        for metric_name, values in metrics.items():
            row[f"mean_{metric_name}"] = np.mean(
                values
            )
            row[f"sd_{metric_name}"] = np.std(
                values
            )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        os.path.join(
            OUT_DIR,
            f"grouped_model_comparison_{tag}.csv",
        ),
        index=False,
    )

    pd.DataFrame(fold_rows).to_csv(
        os.path.join(
            OUT_DIR,
            f"grouped_fold_results_{tag}.csv",
        ),
        index=False,
    )

    print(
        f"\n=== Stratified grouped "
        f"{FOLDS}-fold CV ({tag}) ==="
    )

    display_columns = [
        "model",
        "mean_auc",
        "sd_auc",
        "mean_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1",
    ]

    print(
        comparison[display_columns]
        .round(3)
        .to_string(index=False)
    )

    return comparison, models


#SHAP

def run_tree_shap(model, name, X, tag):
    """
    Fits a tree model on the full analysis dataset and produces
    SHAP feature importance. This is explanatory only; predictive
    performance is taken from grouped cross-validation.
    """

    fitted = clone(model).fit(X, tag[1])
    X_fit = X

    explainer = shap.TreeExplainer(fitted)
    shap_values = explainer.shap_values(X_fit)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, 1]

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_fit,
        show=False,
    )

    plot_path = os.path.join(
        OUT_DIR,
        f"shap_summary_{safe_name}_{tag[0]}.png",
    )

    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    importance = pd.DataFrame(
        {
            "feature": X_fit.columns,
            "mean_abs_shap": np.abs(
                shap_values
            ).mean(axis=0),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.round(4).to_csv(
        os.path.join(
            OUT_DIR,
            f"shap_importance_{safe_name}_{tag[0]}.csv",
        ),
        index=False,
    )


#MAIN

def main():
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    np.random.seed(SEED)

    data = build_pooled()
    data = add_features(data)
    data = add_llm_features(data)

    X_full = build_X(data)

    data.join(
        X_full,
        rsuffix="_feature",
    ).to_csv(
        os.path.join(
            OUT_DIR,
            "pooled_feature_table.csv",
        ),
        index=False,
    )

    for outcome_tag in ["primary", "strict"]:
        y, mask = make_outcome(
            data,
            outcome_tag,
        )

        X = (
            X_full.loc[mask]
            .reset_index(drop=True)
        )

        groups = (
            data.loc[mask, "id"]
            .reset_index(drop=True)
        )

        y = y.reset_index(drop=True)

        print(
            f"\n########## OUTCOME: "
            f"{outcome_tag} "
            f"(n={len(y)}, "
            f"events={int(y.sum())}, "
            f"rate={y.mean():.1%}) "
            f"##########"
        )

        logistic_inference(
            X,
            y,
            outcome_tag,
        )

        comparison, models = (
            grouped_model_comparison(
                X,
                y,
                groups,
                outcome_tag,
            )
        )

        # SHAP for both tree-based comparison models.
        for tree_name in [
            "Random Forest",
            "Gradient Boosting",
        ]:
            fitted_tree = clone(
                models[tree_name]
            ).fit(X, y)

            explainer = shap.TreeExplainer(
                fitted_tree
            )

            shap_values = explainer.shap_values(
                X
            )

            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif getattr(
                shap_values,
                "ndim",
                2,
            ) == 3:
                shap_values = shap_values[:, :, 1]

            safe_name = (
                tree_name.lower()
                .replace(" ", "_")
            )

            importance = pd.DataFrame(
                {
                    "feature": X.columns,
                    "mean_abs_shap": np.abs(
                        shap_values
                    ).mean(axis=0),
                }
            ).sort_values(
                "mean_abs_shap",
                ascending=False,
            )

            importance.round(4).to_csv(
                os.path.join(
                    OUT_DIR,
                    f"shap_importance_{safe_name}_{outcome_tag}.csv",
                ),
                index=False,
            )

            plt.figure()

            shap.summary_plot(
                shap_values,
                X,
                show=False,
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUT_DIR,
                    f"shap_summary_{safe_name}_{outcome_tag}.png",
                ),
                dpi=200,
                bbox_inches="tight",
            )

            plt.close()

    print(
        f"\nAll outputs written to:\n{OUT_DIR}\n"
    )


if __name__ == "__main__":
    main()
