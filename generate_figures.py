"""
generate_figures.py — regenerates every chart used in README.md and the
notebook, plus a metrics.json snapshot of every number quoted in the
write-up. Run after train_model.py (needs churn_production_model.pkl).
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from train_model import (
    CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, load_and_clean,
)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

sns.set_style("whitegrid")
PALETTE = {"No": "#4C72B0", "Yes": "#C44E52"}


def main():
    df = load_and_clean()
    metrics = {}
    metrics["n_customers"] = len(df)
    metrics["churn_rate"] = round((df["Churn"] == "Yes").mean() * 100, 1)

    # --- churn by contract -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    contract_churn = contract_churn.reindex(["Month-to-month", "One year", "Two year"])
    contract_churn.plot(kind="bar", ax=ax, color=[PALETTE["No"], PALETTE["Yes"]])
    ax.set_ylabel("% of customers")
    ax.set_title("Churn rate by contract type")
    ax.legend(title="Churn")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ASSETS / "churn_by_contract.png", dpi=140)
    plt.close(fig)
    counts = pd.crosstab(df["Contract"], df["Churn"])
    metrics["contract"] = {
        c: {
            "churned": int(counts.loc[c, "Yes"]),
            "total": int(counts.loc[c].sum()),
            "pct": round(counts.loc[c, "Yes"] / counts.loc[c].sum() * 100, 1),
        }
        for c in counts.index
    }

    # --- monthly charges histogram -----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["MonthlyCharges"], bins=40, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Monthly charges (USD)")
    ax.set_ylabel("Customers")
    ax.set_title("Distribution of monthly charges")
    plt.tight_layout()
    plt.savefig(ASSETS / "monthly_charges_histogram.png", dpi=140)
    plt.close(fig)

    # --- churn by add-on service (tech support, online security) ----------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, col in zip(axes, ["TechSupport", "OnlineSecurity"]):
        ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
        ct = ct.reindex(["No", "Yes"])
        ct.plot(kind="bar", ax=ax, color=[PALETTE["No"], PALETTE["Yes"]], legend=(col == "OnlineSecurity"))
        ax.set_title(col)
        ax.set_ylabel("% of customers" if col == "TechSupport" else "")
        ax.set_xlabel("Has service")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(ASSETS / "churn_by_addon.png", dpi=140)
    plt.close(fig)
    addon = {}
    for col in ["TechSupport", "OnlineSecurity", "OnlineBackup"]:
        ct = pd.crosstab(df[col], df["Churn"])
        addon[col] = {
            level: round(ct.loc[level, "Yes"] / ct.loc[level].sum() * 100, 1)
            for level in ct.index
        }
    metrics["addon_churn_pct"] = addon

    # --- correlation heatmap -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df[NUMERIC_FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation, numeric features")
    plt.tight_layout()
    plt.savefig(ASSETS / "correlation_heatmap.png", dpi=140)
    plt.close(fig)
    metrics["correlation"] = {
        "tenure_totalcharges": round(corr.loc["tenure", "TotalCharges"], 2),
        "tenure_monthlycharges": round(corr.loc["tenure", "MonthlyCharges"], 2),
    }

    # --- t-test: monthly charges, churned vs not ---------------------------
    churned = df.loc[df["Churn"] == "Yes", "MonthlyCharges"]
    stayed = df.loc[df["Churn"] == "No", "MonthlyCharges"]
    t_stat, p_val = stats.ttest_ind(churned, stayed, equal_var=False)
    metrics["ttest_monthly_charges"] = {
        "t_stat": round(float(t_stat), 2),
        "p_value": float(p_val),
        "mean_churned": round(float(churned.mean()), 2),
        "mean_stayed": round(float(stayed.mean()), 2),
    }

    # --- chi-square battery on every categorical feature --------------------
    chi2_results = []
    for col in CATEGORICAL_FEATURES:
        table = pd.crosstab(df[col], df["Churn"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        chi2_results.append({"feature": col, "chi2": round(float(chi2), 2), "p_value": float(p)})
    chi2_results.sort(key=lambda r: r["chi2"], reverse=True)
    metrics["chi2"] = chi2_results

    # --- model artifact: confusion matrix + coefficients --------------------
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    cm = np.array(artifact["metrics"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted stay", "Predicted churn"],
                yticklabels=["Actual stay", "Actual churn"], ax=ax)
    ax.set_title(f"Confusion matrix — {artifact['model_name']}")
    plt.tight_layout()
    plt.savefig(ASSETS / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    coefs = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index).head(15)
    coef_df = coef_df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in coef_df["coef"]]
    ax.barh(coef_df["feature"], coef_df["coef"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardized / one-hot encoded)")
    ax.set_title("Top 15 churn drivers — logistic regression coefficients")
    plt.tight_layout()
    plt.savefig(ASSETS / "coefficients.png", dpi=140)
    plt.close(fig)

    metrics["top_positive_coefs"] = (
        coef_df[coef_df["coef"] > 0].sort_values("coef", ascending=False)
        [["feature", "coef"]].round(3).to_dict("records")
    )
    metrics["top_negative_coefs"] = (
        coef_df[coef_df["coef"] < 0].sort_values("coef")
        [["feature", "coef"]].round(3).to_dict("records")
    )
    metrics["model_metrics"] = {
        k: v for k, v in artifact["metrics"].items() if k != "confusion_matrix"
    }
    metrics["model_metrics"]["confusion_matrix"] = cm.tolist()
    metrics["model_name"] = artifact["model_name"]

    with open(HERE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {ASSETS} charts and metrics.json")


if __name__ == "__main__":
    main()
