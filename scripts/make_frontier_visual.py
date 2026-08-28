#!/usr/bin/env python3
"""Generate accuracy-compactness frontier figures comparing baselines to GPopt4 and GPTopt8.

Produces publication-quality figures with logarithmic trie size on the x-axis
and held-out F_1/7 score on the y-axis, rendering arrows from hand-tuned baselines
to optimized parameter profiles.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# Standard short display names matching manuscript figures
SHORT_NAMES: Dict[str, str] = {
    "cssk/cshyphen": "cssk",
    "cs/cshyphen_cstenten": "cs/ctt",
    "cs/cshyphen_ujc": "cs/ujc",
    "cs/wiktionary": "cs",
    "de/wiktionary": "de",
    "de/wortliste": "de/wortliste",
    "el/wiktionary": "el",
    "es/wiktionary": "es",
    "is/hyphenation-is": "is",
    "it/wiktionary": "it",
    "nl/wiktionary": "nl",
    "pl/wiktionary": "pl",
    "pt/wiktionary": "pt",
    "ru/wiktionary": "ru",
    "th/orchid": "th",
    "tr/wiktionary": "tr",
    "uk/wiktionary": "uk",
}

# GPTopt8 vectors:
# cssk: (11534, 0.9292) -> (12781, 0.9822) [NAVY arrow pointing up-right]
# th:   (26966, 0.9663) -> (12497, 0.9833) [GREEN arrow pointing up-left]
GPTOPT8_LABELS = {
    # Left cluster
    "pt":          {"dx": -6, "dy":  3, "ha": "right",  "va": "bottom"},  # (1055, 0.9910)
    "it":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},  # (1158, 0.9976)
    "tr":          {"dx":  6, "dy": -2, "ha": "left",   "va": "top"},     # (1328, 0.9919)
    "es":          {"dx":  7, "dy": -5, "ha": "left",   "va": "top"},     # (1395, 0.9844)
    "uk":          {"dx":  5, "dy":  3, "ha": "left",   "va": "bottom"},  # (1828, 0.9727)
    "el":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},  # (2081, 0.9421)

    # Middle group
    "cs":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},  # (4422, 0.9731)
    "pl":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},  # (4831, 0.9872)
    "cs/ujc":      {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},  # (6843, 0.9873)
    "cs/ctt":      {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},  # (8318, 0.9708)

    # Right cluster:
    # th arrowhead is at (12497, 0.9833), navy cssk arrowhead is at (12781, 0.9822)
    "th":          {"dx": -6, "dy":  4, "ha": "right",  "va": "bottom"},  # th label sits top-left of th arrow tip
    "cssk":        {"dx":  6, "dy":  4, "ha": "left",   "va": "bottom"},  # cssk label sits top-right of navy cssk arrow tip
    "is":          {"dx":  6, "dy":  0, "ha": "left",   "va": "center"},  # (14077, 0.9660)
    "nl":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},  # (21762, 0.9881)
    "de":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},  # (23843, 0.9915)
    "ru":          {"dx":  6, "dy":  0, "ha": "left",   "va": "center"},  # (24845, 0.9504)
    "de/wortliste":{"dx":  6, "dy":  3, "ha": "left",   "va": "bottom"},  # (30020, 0.9881)
}

GPOPT4_LABELS = {
    "pt":          {"dx": -6, "dy":  3, "ha": "right",  "va": "bottom"},
    "it":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},
    "tr":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "es":          {"dx":  6, "dy":  0, "ha": "left",   "va": "center"},
    "uk":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},
    "el":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "cs":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "pl":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "cs/ujc":      {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},
    "cs/ctt":      {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "th":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "cssk":        {"dx":  6, "dy":  0, "ha": "left",   "va": "center"},
    "is":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},
    "nl":          {"dx": -6, "dy":  0, "ha": "right",  "va": "center"},
    "de":          {"dx":  0, "dy":  5, "ha": "center", "va": "bottom"},
    "ru":          {"dx":  6, "dy":  0, "ha": "left",   "va": "center"},
    "de/wortliste":{"dx":  6, "dy":  3, "ha": "left",   "va": "bottom"},
}


def load_dataset_rows(repo_root: Path) -> List[Dict[str, object]]:
    bootstrap_path = repo_root / "results/paper2_revision_analysis_currentci/bootstrap_ci.json"
    gptopt8_root = repo_root / "results/gptopt8"

    if not bootstrap_path.exists():
        raise FileNotFoundError(f"Missing bootstrap baseline data at {bootstrap_path}")

    old_rows = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    rows = []

    for old in old_rows:
        dataset = old["dataset"]
        profile_path = gptopt8_root / dataset / "selected_profile.json"
        gptopt8_f = None
        gptopt8_trie = None

        if profile_path.exists():
            new = json.loads(profile_path.read_text(encoding="utf-8"))
            gptopt8_f = float(new["held_out_test_f17"])
            gptopt8_trie = int(new["held_out_test"]["trie_nodes"])

        rows.append({
            "dataset": dataset,
            "name": SHORT_NAMES.get(dataset, dataset),
            "base_trie": int(old["base_trie"]),
            "base_f": float(old["base_f17"]),
            "gpopt4_trie": int(old["opt_trie"]),
            "gpopt4_f": float(old["opt_f17"]),
            "gptopt8_trie": gptopt8_trie,
            "gptopt8_f": gptopt8_f,
        })
    return rows


def setup_typography() -> None:
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
        plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def render_frontier_graph(
    rows: List[Dict[str, object]],
    endpoint_name: str,
    output_png: Path,
    output_pdf: Path = None,
) -> None:
    setup_typography()
    win_color = "#236b43"      # richer forest green for strict Pareto improvement
    bigger_color = "#1b3a6b"   # deep navy for accuracy gain with larger trie

    is_gptopt8 = "8" in endpoint_name
    trie_key = "gptopt8_trie" if is_gptopt8 else "gpopt4_trie"
    f_key = "gptopt8_f" if is_gptopt8 else "gpopt4_f"
    pos_map = GPTOPT8_LABELS if is_gptopt8 else GPOPT4_LABELS

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=300)
    ax.set_xscale("log")
    ax.set_ylim(0.85, 1.012)
    ax.set_xlim(450, 80000)

    for row in rows:
        bt, bf = row["base_trie"], row["base_f"]
        ot, of = row[trie_key], row[f_key]
        if ot is None or of is None:
            continue

        col = win_color if ot < bt else bigger_color

        arr = FancyArrowPatch(
            (bt, bf), (ot, of),
            arrowstyle="-|>", mutation_scale=9.5,
            lw=1.35, color=col, alpha=0.88, zorder=3,
            shrinkA=3.0, shrinkB=3.0,
        )
        ax.add_patch(arr)
        ax.plot([bt], [bf], "o", mfc="white", mec=col, mew=1.1, ms=4.2, zorder=3, alpha=0.88)

        name = row["name"]
        cfg = pos_map.get(name, {"dx": 5, "dy": 5, "ha": "left", "va": "bottom"})
        ax.annotate(
            name, (ot, of),
            xytext=(cfg["dx"], cfg["dy"]), textcoords="offset points",
            ha=cfg["ha"], va=cfg["va"], fontsize=8.6, fontweight="medium",
            color=col, zorder=5,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(colors="#444", labelsize=8.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)

    smaller = sum(1 for r in rows if r[trie_key] is not None and r[trie_key] < r["base_trie"])
    valid = sum(1 for r in rows if r[trie_key] is not None)
    larger = valid - smaller

    ax.set_xlabel("pattern trie size (nodes, log scale)  —  left = more compact", fontsize=10.2)
    ax.set_ylabel(r"$F_{1/7}$  (held-out)  —  up = more accurate", fontsize=11)

    legend = [
        Line2D([0], [0], color=win_color, lw=2.0, marker=">", ms=6, label=f"more accurate $+$ smaller trie ({smaller}/{valid})"),
        Line2D([0], [0], color=bigger_color, lw=2.0, marker=">", ms=6, label=f"more accurate $+$ bigger trie ({larger}/{valid})"),
    ]
    ax.legend(
        handles=legend, loc="lower right", bbox_to_anchor=(0.98, 0.02),
        frameon=False, fontsize=8.2, handlelength=2.2, borderpad=0.2,
    )
    ax.text(
        0.06, 0.155,
        f"{endpoint_name} moves {smaller}/{valid} onto\nbetter-$F_{{1/7}}$, smaller-trie points",
        transform=ax.transAxes, fontsize=8.6, va="top", ha="left", color="#333", linespacing=1.3,
    )

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate accuracy-compactness frontier figures")
    parser.add_argument("--repo-root", default=".", help="Root directory of the repository")
    parser.add_argument("--output-dir", default="results/gptopt8", help="Output directory for generated graphs")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    rows = load_dataset_rows(repo_root)

    render_frontier_graph(
        rows,
        endpoint_name="GPTopt8",
        output_png=out_dir / "published_baseline_to_gptopt8_frontier.png",
        output_pdf=out_dir / "published_baseline_to_gptopt8_frontier.pdf",
    )

    render_frontier_graph(
        rows,
        endpoint_name="GPopt4",
        output_png=out_dir / "published_baseline_to_gpopt4_frontier.png",
        output_pdf=out_dir / "published_baseline_to_gpopt4_frontier.pdf",
    )


if __name__ == "__main__":
    main()
