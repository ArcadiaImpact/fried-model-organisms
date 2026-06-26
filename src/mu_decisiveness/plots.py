from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sentiment_ranking(items, mu, sigma, path):
    order = np.argsort(mu)
    df = pd.DataFrame(
        {"item": [items[i] for i in order], "mu": mu[order], "sigma": sigma[order]}
    )
    plt.figure(figsize=(8, 9))
    ax = sns.barplot(data=df, y="item", x="mu", color="#4C72B0")
    ax.errorbar(df["mu"], range(len(df)), xerr=df["sigma"], fmt="none", ecolor="gray", capsize=2)
    ax.set_xlabel("Thurstonian utility mu (sentiment)")
    ax.set_ylabel("")
    ax.set_title("Gemma-3-12B sentiment ranking")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_validation_scatter(logprob_p_a, gen_p_a, path):
    df = pd.DataFrame({"logprob_p_a": logprob_p_a, "gen_p_a": gen_p_a}).dropna()
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=df, x="logprob_p_a", y="gen_p_a")
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    r = df.corr().iloc[0, 1] if len(df) > 1 else float("nan")
    ax.set_title(f"Logprob vs generation P(pick A)  (r={r:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_preference_heatmap(items, pref, mu, path):
    order = np.argsort(-mu)
    matrix = pref[np.ix_(order, order)]
    labels = [items[i] for i in order]
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="RdBu_r",
        cbar_kws={"label": "P(row > col)"},
    )
    plt.title("Pairwise preference matrix (sorted by mu)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
