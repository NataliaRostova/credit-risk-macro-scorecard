"""Commercial Bank Corporate/SME Credit Risk Scorecard, Validation & Macro Stress Testing.

Pipeline:
1. Synthesizes realistic commercial bank borrower financial dataset (Train / Test / Out-of-Time cohorts)
2. Performs WOE encoding & IV feature selection (IV >= 0.05)
3. Trains L2-regularized Logistic Regression & maps log-odds to standard PDO scorecard points
   (Base Score = 600 at Odds 1:15, PDO = 20)
4. Validates with ROC-AUC, KS Statistic, Gini, Top-20% Bad Capture Lift, and OOT PSI
5. Runs IFRS 9 Expected Credit Loss (ECL = PD * LGD * EAD) Macro Stress Testing
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.woe_iv_engine import WoeIvBinner, compute_psi

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
REPORTS_DIR = ROOT / "reports"

FEATURES = [
    "debt_to_asset_pct",
    "interest_coverage",
    "current_ratio",
    "ocf_to_debt",
    "roe_pct",
    "credit_utilization_pct",
    "revenue_yoy_pct",
    "delinquency_count_12m",
]


def generate_credit_dataset(n_dev: int = 5000, n_oot: int = 1500, seed: int = 202611) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic SME / corporate borrower financial features with true credit risk structure."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def _draw(n: int, macro_shift: float = 0.0) -> pd.DataFrame:
        debt_to_asset = np.clip(rng.normal(58.0 + 2.5 * macro_shift, 14.5, n), 12.0, 95.0)
        interest_cov = np.clip(rng.lognormal(1.25 - 0.12 * macro_shift, 0.55, n), 0.2, 22.0)
        current_ratio = np.clip(rng.lognormal(0.32, 0.35, n), 0.35, 5.0)
        ocf_to_debt = np.clip(rng.normal(0.24 - 0.03 * macro_shift, 0.14, n), -0.30, 0.90)
        roe = np.clip(rng.normal(10.5 - 1.0 * macro_shift, 7.5, n), -25.0, 38.0)
        utilization = np.clip(rng.beta(2.4, 2.8, n) * 100.0, 2.0, 99.0)
        rev_yoy = np.clip(rng.normal(8.5 - 2.0 * macro_shift, 14.0, n), -40.0, 65.0)
        delinq = rng.poisson(0.35 + 0.10 * macro_shift, n)
        ead_mn = np.round(rng.lognormal(2.1, 0.65, n), 2)  # Exposure at Default (Million RMB)

        # Latent structural default log-odds (calibrated to ~6-8% SME credit portfolio default rate)
        logit = (
            -2.15
            + 0.042 * (debt_to_asset - 55.0)
            - 0.38 * np.log1p(interest_cov)
            - 0.48 * (current_ratio - 1.3)
            - 3.40 * ocf_to_debt
            - 0.045 * roe
            + 0.026 * (utilization - 45.0)
            - 0.015 * rev_yoy
            + 0.72 * np.minimum(delinq, 4)
            + rng.normal(0, 0.55, n)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        default = (rng.uniform(0, 1, n) < prob).astype(int)

        return pd.DataFrame(
            {
                "debt_to_asset_pct": np.round(debt_to_asset, 2),
                "interest_coverage": np.round(interest_cov, 2),
                "current_ratio": np.round(current_ratio, 2),
                "ocf_to_debt": np.round(ocf_to_debt, 3),
                "roe_pct": np.round(roe, 2),
                "credit_utilization_pct": np.round(utilization, 2),
                "revenue_yoy_pct": np.round(rev_yoy, 2),
                "delinquency_count_12m": delinq,
                "ead_mn_rmb": ead_mn,
                "default_12m": default,
            }
        )

    dev_df = _draw(n_dev, macro_shift=0.0)
    oot_df = _draw(n_oot, macro_shift=0.35)  # Mild macro softening in OOT cohort
    dev_df.to_csv(DATA_DIR / "credit_dev_cohort.csv", index=False)
    oot_df.to_csv(DATA_DIR / "credit_oot_cohort.csv", index=False)
    return dev_df, oot_df


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_arr = tpr - fpr
    idx = int(np.argmax(ks_arr))
    return float(ks_arr[idx]), float(thresholds[idx]), fpr, tpr, thresholds


def run_scorecard_and_stress_test() -> dict:
    dev_df, oot_df = generate_credit_dataset()
    train_df, test_df = train_test_split(dev_df, test_size=0.30, random_state=42, stratify=dev_df["default_12m"])

    binner = WoeIvBinner(n_bins=5).fit(train_df, FEATURES, "default_12m")
    iv_ranked = sorted(binner.iv_scores_.items(), key=lambda kv: kv[1], reverse=True)
    selected_features = [f for f, iv in iv_ranked if iv >= 0.05]

    X_train = binner.transform(train_df, selected_features)
    y_train = train_df["default_12m"].to_numpy()
    X_test = binner.transform(test_df, selected_features)
    y_test = test_df["default_12m"].to_numpy()
    X_oot = binner.transform(oot_df, selected_features)
    y_oot = oot_df["default_12m"].to_numpy()

    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    clf.fit(X_train, y_train)

    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    p_oot = clf.predict_proba(X_oot)[:, 1]

    # Standard PDO Scorecard Scaling: Score = Offset - Factor * ln(p / (1 - p))
    # Higher score = lower default risk
    pdo = 20.0
    base_score = 600.0
    base_odds = 1.0 / 15.0  # Bad:Good = 1:15
    factor = pdo / np.log(2.0)
    offset = base_score + factor * np.log(base_odds)

    def prob_to_score(p: np.ndarray) -> np.ndarray:
        p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
        log_odds = np.log(p_clip / (1.0 - p_clip))
        return np.round(offset - factor * log_odds, 1)

    s_train = prob_to_score(p_train)
    s_test = prob_to_score(p_test)
    s_oot = prob_to_score(p_oot)

    auc_train = float(roc_auc_score(y_train, p_train))
    auc_test = float(roc_auc_score(y_test, p_test))
    auc_oot = float(roc_auc_score(y_oot, p_oot))
    ks_test, _, fpr_test, tpr_test, _ = compute_ks(y_test, p_test)
    ks_oot, _, fpr_oot, tpr_oot, _ = compute_ks(y_oot, p_oot)
    psi_oot = compute_psi(s_train, s_oot)

    # Top-20% high-risk decile bad capture rate on OOT
    oot_eval = pd.DataFrame({"y": y_oot, "p": p_oot}).sort_values("p", ascending=False)
    top20_n = int(len(oot_eval) * 0.20)
    top20_capture = float(oot_eval.iloc[:top20_n]["y"].sum() / max(oot_eval["y"].sum(), 1) * 100.0)

    # --- IFRS 9 Expected Credit Loss (ECL = PD * LGD * EAD) Macro Stress Testing ---
    # Applied on OOT active loan portfolio (Base LGD = 45%)
    scenarios = [
        {"name": "Baseline (Current)", "rate_bps": 0, "rev_shock": 0.0, "debt_add": 0.0, "lgd": 0.45},
        {"name": "Adverse (+100bps / -8% Rev)", "rate_bps": 100, "rev_shock": -8.0, "debt_add": 3.5, "lgd": 0.50},
        {"name": "Severely Adverse (+200bps / -18% Rev)", "rate_bps": 200, "rev_shock": -18.0, "debt_add": 7.5, "lgd": 0.55},
    ]

    stress_results = []
    total_ead = float(oot_df["ead_mn_rmb"].sum())
    for sc in scenarios:
        shocked = oot_df.copy()
        shocked["debt_to_asset_pct"] = np.clip(shocked["debt_to_asset_pct"] + sc["debt_add"], 12.0, 98.0)
        shocked["interest_coverage"] = np.clip(shocked["interest_coverage"] * (1.0 - sc["rate_bps"] / 550.0), 0.15, 22.0)
        shocked["revenue_yoy_pct"] = shocked["revenue_yoy_pct"] + sc["rev_shock"]
        shocked["ocf_to_debt"] = shocked["ocf_to_debt"] + 0.004 * sc["rev_shock"]

        X_shk = binner.transform(shocked, selected_features)
        pd_shk = clf.predict_proba(X_shk)[:, 1]
        ecl_mn = float(np.sum(pd_shk * sc["lgd"] * shocked["ead_mn_rmb"]))
        stress_results.append(
            {
                "scenario": sc["name"],
                "avg_pd_pct": round(float(pd_shk.mean() * 100.0), 2),
                "lgd_pct": round(sc["lgd"] * 100.0, 1),
                "total_ead_mn_rmb": round(total_ead, 1),
                "portfolio_ecl_mn_rmb": round(ecl_mn, 2),
                "ecl_rate_bps": round((ecl_mn / total_ead) * 10000.0, 1),
            }
        )

    summary = {
        "dev_samples": len(dev_df),
        "oot_samples": len(oot_df),
        "dev_default_rate_pct": round(float(dev_df["default_12m"].mean() * 100.0), 2),
        "oot_default_rate_pct": round(float(oot_df["default_12m"].mean() * 100.0), 2),
        "selected_features_count": len(selected_features),
        "iv_ranking": {k: round(v, 4) for k, v in iv_ranked},
        "auc_train": round(auc_train, 4),
        "auc_test": round(auc_test, 4),
        "auc_oot": round(auc_oot, 4),
        "gini_oot": round(2 * auc_oot - 1, 4),
        "ks_test": round(ks_test, 4),
        "ks_oot": round(ks_oot, 4),
        "psi_oot": round(psi_oot, 4),
        "top20_bad_capture_pct": round(top20_capture, 1),
        "stress_testing": stress_results,
    }

    _plot_risk_charts(iv_ranked, fpr_test, tpr_test, auc_test, ks_test, fpr_oot, tpr_oot, auc_oot, ks_oot, s_oot, y_oot, stress_results)
    _write_validation_report(summary, binner, selected_features, clf, factor, offset)
    return summary


def _plot_risk_charts(
    iv_ranked, fpr_test, tpr_test, auc_test, ks_test, fpr_oot, tpr_oot, auc_oot, ks_oot, s_oot, y_oot, stress_results
) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Chart 1: IV Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 4.8))
    names = [k for k, _ in reversed(iv_ranked)]
    vals = [v for _, v in reversed(iv_ranked)]
    bar_colors = ["#1f4e79" if v >= 0.10 else ("#3498db" if v >= 0.05 else "#95a5a6") for v in vals]
    bars = ax.barh(names, vals, color=bar_colors, height=0.62)
    ax.axvline(0.05, color="#e67e22", ls="--", lw=1.3, label="Selection Threshold (IV = 0.05)")
    ax.axvline(0.10, color="#27ae60", ls=":", lw=1.3, label="Medium Predictive Power (IV = 0.10)")
    for b, v in zip(bars, vals):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Credit Risk Feature Selection: Information Value (IV) Ranking", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Information Value (IV)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "iv_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: ROC Curve & KS Discrimination Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0))
    ax1.plot(fpr_test, tpr_test, color="#1f4e79", lw=2.2, label=f"Holdout Test (AUC = {auc_test:.3f})")
    ax1.plot(fpr_oot, tpr_oot, color="#e67e22", lw=2.2, ls="--", label=f"Out-of-Time (AUC = {auc_oot:.3f})")
    ax1.plot([0, 1], [0, 1], color="#95a5a6", ls=":", lw=1.2)
    ax1.set_title("ROC Curve: Test vs Out-of-Time (OOT)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (Good Cumulative %)")
    ax1.set_ylabel("True Positive Rate (Bad Cumulative %)")
    ax1.legend(loc="lower right", fontsize=9.2)

    ks_x = np.linspace(0, 1, len(tpr_oot))
    ks_idx = int(np.argmax(tpr_oot - fpr_oot))
    ax2.plot(ks_x, tpr_oot, color="#c0392b", lw=2.0, label="Cumulative Default (Bad)")
    ax2.plot(ks_x, fpr_oot, color="#27ae60", lw=2.0, label="Cumulative Non-Default (Good)")
    ax2.vlines(ks_x[ks_idx], fpr_oot[ks_idx], tpr_oot[ks_idx], color="#8e44ad", lw=2.4, ls="--", label=f"Max KS (OOT) = {ks_oot:.3f}")
    ax2.set_title("Kolmogorov-Smirnov (KS) Separation on OOT Cohort", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Score Decile Depth")
    ax2.set_ylabel("Cumulative Proportion")
    ax2.legend(loc="lower right", fontsize=9.2)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "roc_ks_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 3: Scorecard Score Distribution (Good vs Bad) & Macro Stress Test ECL
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax1.hist(s_oot[y_oot == 0], bins=24, density=True, alpha=0.55, color="#27ae60", label="Non-Default (Good)")
    ax1.hist(s_oot[y_oot == 1], bins=24, density=True, alpha=0.60, color="#c0392b", label="Default (Bad)")
    ax1.set_title("OOT Credit Score Distribution (PDO=20, Base=600)", fontsize=11.8, fontweight="bold")
    ax1.set_xlabel("Credit Scorecard Points")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left", fontsize=9)

    sc_labels = [s["scenario"].split("(")[0].strip() for s in stress_results]
    ecls = [s["portfolio_ecl_mn_rmb"] for s in stress_results]
    pds = [s["avg_pd_pct"] for s in stress_results]
    bars = ax2.bar(sc_labels, ecls, color=["#2980b9", "#f39c12", "#c0392b"], width=0.52)
    for b, e, p in zip(bars, ecls, pds):
        ax2.text(b.get_x() + b.get_width() / 2, e + max(ecls) * 0.02, f"¥{e:.1f}M\n(PD {p:.2f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(ecls) * 1.22)
    ax2.set_title("IFRS 9 Macro Stress Testing: Portfolio Expected Credit Loss (ECL)", fontsize=11.8, fontweight="bold")
    ax2.set_ylabel("Expected Credit Loss (Million RMB)")

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "score_distribution_ecl.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_validation_report(summary: dict, binner: WoeIvBinner, selected_features: list[str], clf: LogisticRegression, factor: float, offset: float) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stress_rows = "\n".join(
        f"| {s['scenario']} | {s['avg_pd_pct']}% | {s['lgd_pct']}% | {s['total_ead_mn_rmb']:,.1f} | **{s['portfolio_ecl_mn_rmb']:,.2f}** | {s['ecl_rate_bps']} bps |"
        for s in summary["stress_testing"]
    )
    iv_rows = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in summary["iv_ranking"].items())

    md = f"""# 商业银行信用风险评分卡与 IFRS 9 宏观压力测试验证报告

## 一、模型验证核心指标（Model Validation Scorecard）

| 验证维度 | 指标 | 实测结果 | 银行业风控标准参考 |
| :--- | :--- | :---: | :--- |
| 区分度（Discrimination） | Holdout Test ROC-AUC | **{summary['auc_test']}** | > 0.75（优秀） |
| 跨时间泛化（OOT Generalization） | OOT ROC-AUC / Gini | **{summary['auc_oot']} / {summary['gini_oot']}** | AUC > 0.72 |
| 风险分离度 | Test KS / OOT KS | **{summary['ks_test']} / {summary['ks_oot']}** | KS > 0.35（强分离） |
| 群体稳定性 | OOT PSI | **{summary['psi_oot']}** | < 0.10（高度稳定） |
| 尾部风险捕获 | Top 20% 高风险组违约捕获率 | **{summary['top20_bad_capture_pct']}%** | > 50% |

---

## 二、特征信息价值（IV）排序

| 财务 / 信用特征 | IV 值 |
| :--- | :---: |
{iv_rows}

---

## 三、IFRS 9 宏观压力测试（ECL = PD × LGD × EAD）

| 宏观压力情景 | 组合平均 PD | 假设 LGD | 总敞口 EAD (百万元) | 预期信用损失 ECL (百万元) | 损失率 (bps) |
| :--- | :---: | :---: | :---: | :---: | :---: |
{stress_rows}
"""
    (REPORTS_DIR / "risk_model_validation_report.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    res = run_scorecard_and_stress_test()
    print("[DONE] Credit Risk Scorecard Validation Summary:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
