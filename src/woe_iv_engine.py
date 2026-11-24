"""WOE (Weight of Evidence) & IV (Information Value) Binning Engine + PSI Calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd


class WoeIvBinner:
    """Quantile-based WOE encoder and IV feature selector for credit risk scorecards."""

    def __init__(self, n_bins: int = 5, eps: float = 1e-6) -> None:
        self.n_bins = n_bins
        self.eps = eps
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict[int, float]] = {}
        self.iv_scores_: dict[str, float] = {}
        self.bin_tables_: dict[str, pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame, features: list[str], target: str) -> "WoeIvBinner":
        y = df[target].astype(int)
        total_bad = max(int(y.sum()), 1)
        total_good = max(int((1 - y).sum()), 1)

        for col in features:
            x = df[col].astype(float)
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.unique(np.quantile(x, quantiles))
            if len(edges) < 3:
                edges = np.linspace(x.min(), x.max() + 1e-6, 4)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[col] = edges

            bins = pd.cut(x, bins=edges, labels=False, include_lowest=True)
            tmp = pd.DataFrame({"bin": bins, "y": y})
            grp = tmp.groupby("bin", observed=False)["y"].agg(["count", "sum"]).reset_index()
            grp["good"] = grp["count"] - grp["sum"]
            grp["bad"] = grp["sum"]
            grp["dist_good"] = (grp["good"] + self.eps) / total_good
            grp["dist_bad"] = (grp["bad"] + self.eps) / total_bad
            grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
            grp["iv_component"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
            grp["bad_rate_pct"] = (grp["bad"] / grp["count"].clip(lower=1)) * 100.0

            labels = []
            for b in grp["bin"]:
                lo, hi = edges[int(b)], edges[int(b) + 1]
                lo_s = "-inf" if np.isneginf(lo) else f"{lo:.2f}"
                hi_s = "+inf" if np.isposinf(hi) else f"{hi:.2f}"
                labels.append(f"({lo_s}, {hi_s}]")
            grp["interval"] = labels

            self.woe_maps_[col] = dict(zip(grp["bin"].astype(int), grp["woe"].astype(float)))
            self.iv_scores_[col] = float(grp["iv_component"].sum())
            self.bin_tables_[col] = grp[["bin", "interval", "count", "bad", "bad_rate_pct", "woe", "iv_component"]]
        return self

    def transform(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in features:
            bins = pd.cut(df[col].astype(float), bins=self.bin_edges_[col], labels=False, include_lowest=True)
            out[col] = bins.map(self.woe_maps_[col]).fillna(0.0).astype(float)
        return out


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between development and OOT score distributions."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_cnt = np.histogram(expected, bins=edges)[0] / len(expected)
    act_cnt = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_cnt = np.clip(exp_cnt, 1e-6, None)
    act_cnt = np.clip(act_cnt, 1e-6, None)
    return float(np.sum((act_cnt - exp_cnt) * np.log(act_cnt / exp_cnt)))
