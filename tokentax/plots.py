"""Charts + JSON for the web demo, from results/results.csv.

    python -m tokentax.plots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

INDIC = ["Hindi", "Bengali", "Marathi", "Gujarati", "Punjabi", "Odia",
         "Assamese", "Tamil", "Telugu", "Kannada", "Malayalam", "Urdu"]
LOSSY = 0.05  # unk_rate above this: the tokenizer dropped text, so the tax is not comparable

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def indic_mean(df: pd.DataFrame) -> pd.Series:
    rows = df[df.language.isin(INDIC) & (df.unk_rate <= LOSSY)]
    return rows.groupby("tokenizer").tax.mean().sort_values()


def heatmap(df: pd.DataFrame, out: Path):
    order_t = indic_mean(df).index.tolist()
    order_l = df[df.language != "English"].groupby("language").tax.median().sort_values().index.tolist()
    grid = df.pivot(index="language", columns="tokenizer", values="tax").loc[order_l, order_t]
    unk = df.pivot(index="language", columns="tokenizer", values="unk_rate").loc[order_l, order_t]

    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(grid.values, cmap=cmap, norm=LogNorm(vmin=1, vmax=12.5), aspect="auto")
    for i, lang in enumerate(order_l):
        for j, tname in enumerate(order_t):
            v = grid.iloc[i, j]
            lossy = unk.iloc[i, j] > LOSSY
            label = f"{v:.1f}" + ("*" if lossy else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if v >= 3.2 else INK)
    ax.set_xticks(range(len(order_t)), order_t, rotation=30, ha="right", color=INK)
    ax.set_yticks(range(len(order_l)), order_l, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Token tax: tokens for the same sentence vs English (1.0 = same cost)",
                 loc="left", color=INK, fontsize=12, pad=12)
    fig.text(0.01, 0.01, "FLORES-200 devtest, 1012 parallel sentences.  * = over 5% of tokens are [UNK]: "
             "text was dropped, not compressed.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def bars(df: pd.DataFrame, out: Path):
    m = indic_mean(df)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(m.index, m.values, color="#2a78d6", height=0.6)
    for y, v in enumerate(m.values):
        ax.text(v + 0.08, y, f"{v:.2f}x", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()  # best on top
    ax.axvline(1, color=INK2, lw=1, ls=":")
    ax.set_xlabel("average token tax over 12 Indian languages (lower is better)", color=INK2)
    ax.tick_params(colors=INK, length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.set_title("Which tokenizer charges Indian languages the least?", loc="left", color=INK, fontsize=12)
    fig.text(0.01, 0.01, f"Cells with over {LOSSY:.0%} [UNK] tokens are left out (mBERT on Odia). Dotted line = English.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def export_json(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lossy_threshold": LOSSY, "rows": df.to_dict("records")}, ensure_ascii=False),
                   encoding="utf-8")


def main(csv=Path("results/results.csv")):
    df = pd.read_csv(csv)
    heatmap(df, csv.parent / "heatmap.png")
    bars(df, csv.parent / "by_tokenizer.png")
    export_json(df, Path("docs/results.json"))


if __name__ == "__main__":
    main()
