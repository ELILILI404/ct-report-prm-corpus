"""Score a Best-of-N selection run with CRG, the paper's clinical metric.

CRG compares an 18-class abnormality-label vector extracted by the
RadBERT-based classifier released with CT-CHAT (one vector for the selected
report, one for the reference report) and combines them as

    w    = (T - A) / (2A)
    Smax = A * w
    s    = w*TP - w*FN - FP
    CRG  = Smax / (2*Smax - s)

where T is the number of classes (with negatives), A = TP + FN is the number
of reference-positive classes, and TP/FP/FN are counted over the 18 classes
for one scan. This is the identical formula used to produce every CRG number
in the paper and in this repository's reproduce/results_table1.csv.

This script does not run the RadBERT classifier itself -- that step is
released with CT-CHAT (see the top-level README, "Not included"). It expects
two CSVs already produced by that classifier, one row per scan, indexed by
AccessionNo, with one 0/1 (or continuous, rounded here) column per class in
pipeline/common.py's CLASSES:

    --gt-labels    reference-report labels  (one row per scan)
    --pred-labels  selected-report labels   (one row per scan)

and the selected.jsonl written by 10_select_bestofn.py, used only to restrict
scoring to the scans that were actually selected (so a partial run can be
scored without editing the label CSVs).

Series de-duplication (--dedup-series): CT-RATE ships up to three
reconstructions per patient-series. The paper's headline CRG (Table 1) is
computed on one volume per patient-series (1,564 cases from the 3,039-volume
validation split), keeping the first reconstruction seen after sorting by
reconstruction index, matching AccessionNo of the form
"valid_{patient}_{series}_{reconstruction}". Passing this flag reproduces
that de-duplication from the label CSVs directly; the exact 1,564-case list
used for the paper is not included in this repository (see README).

Usage:
    11_score_crg.py --gt-labels ref_labels.csv --pred-labels selected_labels.csv \
                     --selected selected.jsonl [--dedup-series] \
                     [--bootstrap 10000] [--seed 42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES  # noqa: E402


def paper_crg(yt: np.ndarray, yp: np.ndarray) -> float:
    """CRG for one or more scans, given true/predicted 0/1 label matrices."""
    yt = np.asarray(yt).astype(int)
    yp = np.asarray(yp).astype(int)
    TP = int(((yt == 1) & (yp == 1)).sum())
    FP = int(((yt == 0) & (yp == 1)).sum())
    FN = int(((yt == 1) & (yp == 0)).sum())
    TN = int(((yt == 0) & (yp == 0)).sum())
    T, A = TP + FP + FN + TN, TP + FN
    if A == 0:
        return float("nan")
    w = (T - A) / (2 * A)
    Smax = A * w
    s = TP * w - FN * w - FP
    denom = 2 * Smax - s
    return Smax / denom if denom != 0 else float("nan")


def series_key(acc: str) -> str:
    parts = acc.split("_")
    return "_".join(parts[:3]) if len(parts) >= 4 else acc


def recon_index(acc: str) -> int:
    try:
        return int(acc.split("_")[-1])
    except ValueError:
        return 999


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("AccessionNo")
    missing = [c for c in CLASSES if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing class columns {missing}")
    return df[CLASSES].clip(0, 1).round().astype(int)


def bootstrap_ci(gt: pd.DataFrame, pred: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """Scan-level bootstrap: resample scans with replacement, recompute CRG."""
    rng = np.random.default_rng(seed)
    n = len(gt)
    idx = gt.index.to_numpy()
    scores = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(idx, size=n, replace=True)
        scores[i] = paper_crg(gt.loc[pick].values, pred.loc[pick].values)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-labels", required=True, help="reference-report label CSV, indexed by AccessionNo")
    ap.add_argument("--pred-labels", required=True, help="selected-report label CSV, indexed by AccessionNo")
    ap.add_argument("--selected", help="selected.jsonl from 10_select_bestofn.py; restricts scoring to these volume_ids")
    ap.add_argument("--dedup-series", action="store_true",
                     help="keep one reconstruction per patient-series (paper's 1,564-case protocol)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples for the 95%% CI (0 to skip)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    gt, pred = load_labels(a.gt_labels), load_labels(a.pred_labels)
    common = gt.index.intersection(pred.index)

    if a.selected:
        wanted = {json.loads(l)["volume_id"] for l in open(a.selected, encoding="utf-8")}
        common = common.intersection(wanted)

    if len(common) == 0:
        raise SystemExit("no overlapping AccessionNo between the two label files (and --selected, if given)")

    gt, pred = gt.loc[common], pred.loc[common]
    print(f"scans scored: {len(common)}")

    if a.dedup_series:
        meta = pd.DataFrame({"acc": common})
        meta["series"] = meta["acc"].map(series_key)
        meta["recon"] = meta["acc"].map(recon_index)
        keep = (meta.sort_values(["series", "recon"])
                    .groupby("series", as_index=False).first()["acc"])
        gt, pred = gt.loc[keep], pred.loc[keep]
        print(f"after series de-duplication: {len(keep)} cases")

    crg = paper_crg(gt.values, pred.values)
    print(f"CRG = {crg:.4f}")

    if a.bootstrap > 0:
        lo, hi = bootstrap_ci(gt, pred, a.bootstrap, a.seed)
        print(f"95% CI [{lo:.4f}, {hi:.4f}]  ({a.bootstrap} resamples, seed={a.seed})")


if __name__ == "__main__":
    main()
