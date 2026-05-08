"""Generate all figures used by final_report.qmd.

Run from the project root:
    python generate_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict


RANDOM_STATE = 42
COMPETITIVE_MIN = 0.30
COMPETITIVE_MAX = 0.70
FIPS_TO_DIVISION = {
    9: 1, 23: 1, 25: 1, 33: 1, 44: 1, 50: 1,
    34: 2, 36: 2, 42: 2,
    17: 3, 18: 3, 26: 3, 39: 3, 55: 3,
    19: 4, 20: 4, 27: 4, 29: 4, 31: 4, 38: 4, 46: 4,
    10: 5, 11: 5, 12: 5, 13: 5, 24: 5, 37: 5, 45: 5, 51: 5, 54: 5,
    1: 6, 21: 6, 28: 6, 47: 6,
    5: 7, 22: 7, 40: 7, 48: 7,
    4: 8, 8: 8, 16: 8, 30: 8, 32: 8, 35: 8, 49: 8, 56: 8,
    2: 9, 6: 9, 15: 9, 41: 9, 53: 9,
}
DIVISION_FEATURES = [f"division_{i}" for i in range(2, 10)]
MODEL_SPECS = {
    "M1": {
        "title": "Model 1\n(Age + Gender + Division)",
        "label": "Model 1 (Age + Gender + Division)",
        "features": ["age", "gender_bin", *DIVISION_FEATURES],
        "color": "#5B9BD5",
        "line": ":",
    },
    "M2": {
        "title": "Model 2\n(+ Race)",
        "label": "Model 2 (+ Race)",
        "features": ["age", "gender_bin", *DIVISION_FEATURES, "race_r"],
        "color": "#70AD47",
        "line": "--",
    },
    "M3": {
        "title": "Model 3\n(+ Education, Income & More)",
        "label": "Model 3 (+ Education, Income & More)",
        "features": [
            "age",
            "gender_bin",
            *DIVISION_FEATURES,
            "race_r",
            "educ",
            "income_r",
            "marstat_r",
            "employ_r",
            "ownhome",
            "investor",
        ],
        "color": "#C0392B",
        "line": "-",
    },
}

FEATURE_LABELS = {
    "age": "Age",
    "gender_bin": "Gender",
    "division": "Census Division",
    "race_r": "Race",
    "educ": "Education",
    "income_r": "Family Income",
    "marstat_r": "Marital Status",
    "employ_r": "Employment",
    "ownhome": "Home Ownership",
    "investor": "Stock Investor",
}

RACE_LABELS = {
    1: "White",
    2: "Black",
    3: "Hispanic",
    4: "Asian",
    5: "Other",
}

EDUC_LABELS = {
    1: "No HS",
    2: "HS Grad",
    3: "Some College",
    4: "2-yr Degree",
    5: "4-yr Degree",
    6: "Postgrad",
}


def load_data():
    cols = [
        "birthyr",
        "gender4",
        "race",
        "educ",
        "faminc_new",
        "marstat",
        "employ",
        "ownhome",
        "investor",
        "inputstate",
        "pid7",
    ]
    raw = pd.read_csv(
        "data/CCES22_Common_OUTPUT_vv_topost.csv", usecols=cols, low_memory=False
    )

    df = raw.copy()
    df.loc[df["faminc_new"] == 97, "faminc_new"] = np.nan
    df["age"] = 2022 - df["birthyr"]
    df["gender_bin"] = df["gender4"].apply(lambda x: 1 if x == 1 else 2)
    df = df[df["pid7"].isin([1, 2, 3, 5, 6, 7])].copy()
    df["target"] = (df["pid7"] <= 3).astype(int)
    df["division"] = df["inputstate"].map(FIPS_TO_DIVISION)
    for division in range(2, 10):
        df[f"division_{division}"] = (df["division"] == division).astype(int)

    # Recoded variables (matching the interactive selector).
    df["race_r"] = df["race"].replace({5: 5, 6: 5, 7: 5})
    income_map = {
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 5,
        12: 6,
        13: 6,
        14: 7,
        15: 8,
        16: 9,
    }
    df["income_r"] = df["faminc_new"].map(income_map)
    df["employ_r"] = df["employ"].map(
        {1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 7}
    )
    df["marstat_r"] = df["marstat"].map({1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5})

    for col in ["income_r", "marstat_r", "employ_r", "ownhome", "investor"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def fit_cross_validated_predictions(df):
    y = df["target"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    probas = {}
    aucs = {}
    for name, spec in MODEL_SPECS.items():
        X = df[spec["features"]].values
        probas[name] = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs[name] = roc_auc_score(y, probas[name])

    return probas, aucs


def save_fig1_prob_distributions(probas, aucs):
    bins = np.linspace(0, 1, 51)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False)
    fig.suptitle(
        "Distribution of Predicted Probability of Democratic Partisanship",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for ax, name in zip(axes, ["M1", "M2", "M3"]):
        scores = probas[name]
        spec = MODEL_SPECS[name]
        ax.axvspan(COMPETITIVE_MIN, COMPETITIVE_MAX, color="#FFF2CC", alpha=0.75)
        ax.axvline(COMPETITIVE_MIN, color="gray", linestyle="--", linewidth=1)
        ax.axvline(COMPETITIVE_MAX, color="gray", linestyle="--", linewidth=1)
        ax.hist(scores, bins=bins, color=spec["color"], edgecolor="white", alpha=0.85)
        swing_share = (
            (scores >= COMPETITIVE_MIN) & (scores <= COMPETITIVE_MAX)
        ).mean()
        ax.text(
            0.50,
            3500,
            f"{swing_share:.0%} in\nswing zone",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF2CC", ec="#D6B300"),
        )
        ax.set_title(f'{spec["title"]}\nAUC = {aucs[name]:.3f}', fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Pr(Democrat)")
        ax.set_ylabel("Number of Respondents")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 4000)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.02,
        'Note: Gold shading marks the "competitive zone" (0.30-0.70) where campaign spending is concentrated.',
        ha="center",
        fontsize=9,
        style="italic",
        color="dimgray",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig("fig1_prob_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fig2_roc_curves(df, probas, aucs):
    y = df["target"].values
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in ["M1", "M2", "M3"]:
        spec = MODEL_SPECS[name]
        fpr, tpr, _ = roc_curve(y, probas[name])
        ax.plot(
            fpr,
            tpr,
            color=spec["color"],
            linestyle=spec["line"],
            linewidth=3,
            label=f'{spec["label"]} — AUC {aucs[name]:.3f}',
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.3, alpha=0.7,
            label="Chance (AUC 0.500)")
    ax.set_title("ROC Curves by Model Specification", fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("fig2_roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fig3_feature_importance(df):
    features = MODEL_SPECS["M3"]["features"]
    y = df["target"].values
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(df[features].values, y)
    importances = pd.Series(rf.feature_importances_, index=features) * 100
    division_importance = importances.reindex(DIVISION_FEATURES).sum()
    importances = importances.drop(labels=DIVISION_FEATURES, errors="ignore")
    importances.loc["division"] = division_importance
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#C0392B" if value >= 7 else "#A9BED3" for value in importances]
    labels = [FEATURE_LABELS[f] for f in importances.index]
    ax.barh(labels, importances.values, color=colors)
    for i, value in enumerate(importances.values):
        ax.text(value + 0.2, i, f"{value:.1f}%", va="center", fontsize=10)
    ax.set_title(
        "Feature Importance — Model 3 (Consumer Data)\n"
        "Predicting Democratic Partisanship, CCES 2022",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Decrease in Impurity (%)")
    ax.set_xlim(0, max(importances.max() * 1.25, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("fig3_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def group_swing_share(df, probas, group_col, label_map):
    rows = []
    for code, label in label_map.items():
        mask = df[group_col] == code
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "label": label,
                "M1": (
                    (probas["M1"][mask] >= COMPETITIVE_MIN)
                    & (probas["M1"][mask] <= COMPETITIVE_MAX)
                ).mean()
                * 100,
                "M3": (
                    (probas["M3"][mask] >= COMPETITIVE_MIN)
                    & (probas["M3"][mask] <= COMPETITIVE_MAX)
                ).mean()
                * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("M3", ascending=True)


def save_fig5_who_is_excluded(df, probas):
    race_df = group_swing_share(df, probas, "race_r", RACE_LABELS)
    educ_df = group_swing_share(df, probas, "educ", EDUC_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    fig.suptitle(
        "Which Voter Groups Remain 'Targetable' Under Each Model?\n"
        "(% of group in competitive zone 0.30-0.70)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for ax, data, title in [
        (axes[0], race_df, "By Race / Ethnicity"),
        (axes[1], educ_df, "By Education Level"),
    ]:
        y_pos = np.arange(len(data))
        height = 0.36
        ax.barh(y_pos + height / 2, data["M1"], height=height, color="#5B9BD5",
                label="Model 1 (Age+Gender+Division)")
        ax.barh(y_pos - height / 2, data["M3"], height=height, color="#C0392B",
                label="Model 3 (Consumer Data)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["label"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("% of Group in Competitive Zone")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig("fig5_who_is_excluded.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_evaluation(df, probas, aucs):
    y = df["target"].values
    matrix_files = {}

    for name in ["M1", "M2", "M3"]:
        preds = (probas[name] >= 0.5).astype(int)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        filename = f"fig_confusion_{name.lower()}.png"
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nRepublican", "Predicted\nDemocrat"])
        ax.set_yticklabels(["Actual\nRepublican", "Actual\nDemocrat"])
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        threshold = cm.max() / 1.5
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                color = "white" if cm[row, col] > threshold else "#1f1f1f"
                ax.text(
                    col,
                    row,
                    f"{cm[row, col]:,}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        matrix_files[name] = filename

    def evaluation_lines(model_names, heading):
        lines = [
            heading,
            "",
            "| Model | Accuracy | AUC |",
            "|---|---:|---:|",
        ]
        for name in model_names:
            preds = (probas[name] >= 0.5).astype(int)
            lines.append(
                f"| {name} | {accuracy_score(y, preds):.3f} | {aucs[name]:.3f} |"
            )
        for name in model_names:
            lines.extend(
                [
                    "",
                    f"#### {name} Confusion Matrix",
                    "",
                    f"![{name} confusion matrix.]({matrix_files[name]})" + "{width=55%}",
                ]
            )
        return lines

    m1_m2_lines = evaluation_lines(
        ["M1", "M2"],
        "### Model Performance for M1 and M2",
    )
    m3_lines = evaluation_lines(
        ["M3"],
        "### Model 3 Performance",
    )

    with open("model_evaluation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(m1_m2_lines) + "\n")

    with open("model_evaluation_m3.md", "w", encoding="utf-8") as f:
        f.write("\n".join(m3_lines) + "\n")


def generate_widget_data(df):
    y = df["target"].values
    widget_data = {"models": {}}

    for name, spec in MODEL_SPECS.items():
        features = spec["features"]
        X = df[features].values.astype(float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_scaled = (X - mean) / std

        logit = LogisticRegression(max_iter=1000)
        logit.fit(X_scaled, y)

        widget_data["models"][name.lower()] = {
            "feats": features,
            "coef": logit.coef_[0].tolist(),
            "intercept": float(logit.intercept_[0]),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    return widget_data


def main():
    df = load_data()
    probas, aucs = fit_cross_validated_predictions(df)
    save_fig1_prob_distributions(probas, aucs)
    save_fig2_roc_curves(df, probas, aucs)
    save_fig3_feature_importance(df)
    save_fig5_who_is_excluded(df, probas)
    save_model_evaluation(df, probas, aucs)
    print(json.dumps(generate_widget_data(df), separators=(",", ":")))


if __name__ == "__main__":
    main()
