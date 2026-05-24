#!/usr/bin/env python3
"""
Parse optimize.py logs from a batch run and emit results.csv + results_table.tex.

Usage:
    python collect_results.py <config.json> [--output-dir DIR]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# --- parsing ---

_RE_PARAMS     = re.compile(r"Best parameters:\s+(.+)")
_RE_BAD_W      = re.compile(r"bad_weights=\(([^)]+)\),\s*threshold=(\d+)")
_RE_RESULTS    = re.compile(r"good=([\d.]+),\s*bad=([\d.]+),\s*missed=([\d.]+)")
_RE_VARIANCE   = re.compile(r"good_variance=([\d.]+),\s*bad_variance=([\d.]+),\s*missed_variance=([\d.]+)")
_RE_PATTERNS   = re.compile(r"n_patterns=([\d.]+),\s*trie_nodes=([\d.]+)")
_RE_SCORE      = re.compile(r"score=([\d.]+)")
# cross_validate.py tabular output: "lang & name & profile & f_score & trie_nodes \\"
_RE_CV_ROW     = re.compile(r"[\w\\_-]+\s*&[^&]+&[^&]+&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\")


def parse_log(log_path: Path) -> dict | None:
    """Extract the final OPTIMIZATION COMPLETE block from a log file."""
    try:
        text = log_path.read_text()
    except FileNotFoundError:
        return None

    # Find the last occurrence of the block
    marker = "OPTIMIZATION COMPLETE"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    block = text[idx:]

    result = {}

    m = _RE_BAD_W.search(block)
    if m:
        result["bad_weights"] = tuple(int(x) for x in m.group(1).split(","))
        result["threshold"] = int(m.group(2))
    else:
        m2 = _RE_PARAMS.search(block)
        result["raw_params"] = m2.group(1).strip() if m2 else ""

    m = _RE_RESULTS.search(block)
    if m:
        result["good"]   = float(m.group(1))
        result["bad"]    = float(m.group(2))
        result["missed"] = float(m.group(3))

    m = _RE_VARIANCE.search(block)
    if m:
        result["good_variance"]   = float(m.group(1))
        result["bad_variance"]    = float(m.group(2))
        result["missed_variance"] = float(m.group(3))

    m = _RE_PATTERNS.search(block)
    if m:
        result["n_patterns"] = int(float(m.group(1)))
        result["trie_nodes"] = float(m.group(2))

    m = _RE_SCORE.search(block)
    if m:
        result["score"] = float(m.group(1))

    return result or None


def parse_crossval_log(log_path: Path) -> dict | None:
    """Extract F1/7 and trie_nodes from cross_validate.py tabular output."""
    try:
        text = log_path.read_text()
    except FileNotFoundError:
        return None

    marker = "CROSS-VALIDATION RESULTS"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    m = _RE_CV_ROW.search(text, idx)
    if not m:
        return None

    return {
        "cv_f17":        float(m.group(1)),
        "cv_trie_nodes": float(m.group(2)),
    }


# --- formatting ---

def _fmt_weights(r: dict) -> str:
    if "bad_weights" in r:
        return str(list(r["bad_weights"]))
    return r.get("raw_params", "?")


def _escape_latex(s: str) -> str:
    return (s.replace("_", r"\_")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#"))


CSV_FIELDS = [
    "name",
    "suk_objective", "suk_args",
    "fuk_objective", "fuk_args",
    "iterations", "batch_size",
    "weight", "annotation",
    # SUK results
    "suk_bad_weights", "suk_threshold",
    "suk_good", "suk_good_var", "suk_bad", "suk_bad_var", "suk_missed", "suk_missed_var",
    "suk_n_patterns", "suk_trie_nodes", "suk_score",
    # FUK results
    "fuk_bad_weights", "fuk_threshold",
    "fuk_good", "fuk_good_var", "fuk_bad", "fuk_bad_var", "fuk_missed", "fuk_missed_var",
    "fuk_n_patterns", "fuk_trie_nodes", "fuk_score",
    # Cross-validation (FUK wordlist, FUK best params)
    "cv_f17", "cv_trie_nodes",
]


def build_row(entry: dict, suk: dict | None, fuk: dict | None, cv: dict | None) -> dict:
    row = {
        "name":          entry["name"],
        "suk_objective": entry["suk_objective"],
        "suk_args":      entry.get("suk_args", ""),
        "fuk_objective": entry["fuk_objective"],
        "fuk_args":      entry.get("fuk_args", ""),
        "iterations":    entry.get("iterations", 30),
        "batch_size":    entry.get("batch_size", 1),
        "weight":        entry.get("weight", ""),
        "annotation":    entry.get("annotation", ""),
    }
    for prefix, res in [("suk", suk), ("fuk", fuk)]:
        objective = entry.get(f"{prefix}_objective", "")
        if res:
            row[f"{prefix}_bad_weights"] = str(list(res.get("bad_weights", [])))
            row[f"{prefix}_threshold"]   = res.get("threshold", "")
            row[f"{prefix}_good"]        = res.get("good", "")
            row[f"{prefix}_bad"]         = res.get("bad", "")
            row[f"{prefix}_missed"]      = res.get("missed", "")
            row[f"{prefix}_n_patterns"]  = res.get("n_patterns", "")
            row[f"{prefix}_trie_nodes"]  = res.get("trie_nodes", "")
            row[f"{prefix}_score"]       = res.get("score", "")
            if objective == "f17_cv":
                row[f"{prefix}_good_var"]   = res.get("good_variance", "N/A")
                row[f"{prefix}_bad_var"]    = res.get("bad_variance", "N/A")
                row[f"{prefix}_missed_var"] = res.get("missed_variance", "N/A")
            else:
                row[f"{prefix}_good_var"] = row[f"{prefix}_bad_var"] = row[f"{prefix}_missed_var"] = ""
        else:
            for f in ["bad_weights","threshold","good","good_var","bad","bad_var",
                      "missed","missed_var","n_patterns","trie_nodes","score"]:
                row[f"{prefix}_{f}"] = "N/A"
    row["cv_f17"]        = cv["cv_f17"]        if cv else "N/A"
    row["cv_trie_nodes"] = cv["cv_trie_nodes"] if cv else "N/A"
    return row


def write_latex(rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabularx}{Xllrrrrrrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Name} & \textbf{Phase} & \textbf{Objective} & \textbf{Bad weights} & "
        r"\textbf{Thr.} & \textbf{Good} & \textbf{Good $\sigma^2$} & "
        r"\textbf{Bad} & \textbf{Bad $\sigma^2$} & "
        r"\textbf{Missed} & \textbf{Missed $\sigma^2$} & "
        r"\textbf{Patterns} & \textbf{Score} & \textbf{CV $F_{1/7}$} & \textbf{CV nodes} \\"
    )
    lines.append(r"\hline")

    for row in rows:
        for prefix, label in [("suk", "SUK"), ("fuk", "FUK")]:
            obj   = "\\mcode{" + _escape_latex(row[f"{prefix}_objective"]) + "}"
            bw    = _escape_latex(str(row[f"{prefix}_bad_weights"]))
            thr   = row[f"{prefix}_threshold"]
            good  = row[f"{prefix}_good"]
            bad   = row[f"{prefix}_bad"]
            miss  = row[f"{prefix}_missed"]
            npat  = row[f"{prefix}_n_patterns"]
            score = row[f"{prefix}_score"]
            name_col = "\\mcode{" + _escape_latex(row["name"]) + "}" if prefix == "suk" else ""
            if prefix == "fuk":
                cv_f17   = row["cv_f17"]
                cv_nodes = row["cv_trie_nodes"]
            else:
                cv_f17 = cv_nodes = ""
            good_var = row[f"{prefix}_good_var"]
            bad_var  = row[f"{prefix}_bad_var"]
            miss_var = row[f"{prefix}_missed_var"]
            lines.append(
                f"{name_col} & {label} & {obj} & {bw} & {thr} & "
                f"{good} & {good_var} & {bad} & {bad_var} & {miss} & {miss_var} & "
                f"{npat} & {score} & {cv_f17} & {cv_nodes} \\\\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabularx}}")
    lines.append(r"\caption{Optimization results}")
    lines.append(r"\label{tab:opt_results}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")


# --- main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON config file used for the batch run")
    parser.add_argument("--output-dir", default=".", help="Where to write results.csv and results_table.tex")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open() as f:
        entries = json.load(f)

    rows = []
    base = Path("/var/tmp/xhulka")

    for entry in entries:
        name = entry["name"]
        suk_log = base / name / "suk" / "optimize.log"
        fuk_log = base / name / "fuk" / "optimize.log"
        cv_log  = base / name / "fuk" / "crossval.log"

        suk = parse_log(suk_log)
        fuk = parse_log(fuk_log)
        cv  = parse_crossval_log(cv_log)

        if suk is None:
            print(f"  [WARN] No SUK results for {name} (log: {suk_log})", file=sys.stderr)
        if fuk is None:
            print(f"  [WARN] No FUK results for {name} (log: {fuk_log})", file=sys.stderr)
        if cv is None:
            print(f"  [WARN] No cross-validation results for {name} (log: {cv_log})", file=sys.stderr)

        rows.append(build_row(entry, suk, fuk, cv))

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV  -> {csv_path}")

    tex_path = out_dir / "results_table.tex"
    write_latex(rows, tex_path)
    print(f"LaTeX -> {tex_path}")


if __name__ == "__main__":
    main()
