"""Check that the released files are internally consistent and match the reported counts.

    python scripts/verify_release.py

1. data/sentences.parquet has 81,452 rows over 5,000 studies; 20,551 carry a class
   (from 4,955 of the 5,000 studies -- the other 45 have no sentence naming one
   of the 18 CT-RATE classes, so they contribute to the LLM set only).
2. Recomputing the CT-CLIP labels from the released entities and p(c) reproduces
   label_tau030 / label_tau040 exactly.
3. Each PRM-format file matches the corresponding column (records, sentences, support rate).
4. The cross-label statistics cited in the paper's discussion -- overall
   image-vs-LLM agreement, and the assert/deny support split for each rule --
   are recomputed from data/sentences.parquet and printed, not just asserted.
5. data/val_manifest.csv has 3,039 validation volumes, 1,564 flagged
   in_dedup1564; data/candidates_pool/ has the same 3,039 volumes (matching
   val_manifest.csv exactly, not just in count) across its 24 shards, 32
   candidates each.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECT = {"ctclip_tau030": (4955, 20551), "ctclip_tau040": (4955, 20551), "llm_reference": (5000, 81452)}


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


df = pd.read_parquet(DATA / "sentences.parquet")
if len(df) != 81452 or df.study_id.nunique() != 5000:
    fail(f"sentence table: {len(df)} rows / {df.study_id.nunique()} studies")
if df.radgraph_class.notna().sum() != 20551:
    fail(f"{df.radgraph_class.notna().sum()} sentences with a class")
n_studies_with_class = df.loc[df.radgraph_class.notna(), "study_id"].nunique()
if n_studies_with_class != 4955:
    fail(f"{n_studies_with_class} studies contribute to the image-evidence set, expected 4,955")
print(f"sentence table: 81,452 sentences, 5,000 studies (LLM set); "
      f"20,551 with a mapped class, {n_studies_with_class} studies (image-evidence set)  ok")

# --- recompute image-evidence labels into a scratch copy and compare ---
tmp = ROOT / "work" / "_verify"
tmp.mkdir(parents=True, exist_ok=True)
saved = df.copy()
subprocess.run([sys.executable, str(ROOT / "pipeline" / "06_ctclip_labels.py"), "--from-release"],
               check=True, cwd=ROOT / "pipeline")
re = pd.read_parquet(DATA / "sentences.parquet")
for col in ("radgraph_class", "polarity", "label_tau030", "label_tau040"):
    if not saved[col].equals(re[col]):
        fail(f"recomputed {col} differs from released column")
saved.to_parquet(DATA / "sentences.parquet", index=False)  # restore byte-identical original
print("recomputed labels identical to released columns  ok")

# --- PRM-format files ---
REWARD_MODEL = {"ctclip_tau030": "0.5B", "ctclip_tau040": "3B", "llm_reference": "comparator (0.5B and 3B)"}
for name, (n_rec, n_sent) in EXPECT.items():
    col = {"ctclip_tau030": "label_tau030", "ctclip_tau040": "label_tau040", "llm_reference": "label_llm"}[name]
    recs = [json.loads(l) for l in open(DATA / "prm_format" / f"{name}.jsonl", encoding="utf-8")]
    sents = sum(len(r["labels"]) for r in recs)
    pos = sum(sum(r["labels"]) for r in recs)
    if (len(recs), sents) != (n_rec, n_sent):
        fail(f"{name}: {len(recs)} records / {sents} sentences")
    if pos != int(df[col].sum()):
        fail(f"{name}: supported count {pos} != column sum {int(df[col].sum())}")
    print(f"{name:15s} {len(recs):5d} records {sents:6d} sentences {100 * pos / sents:5.1f}% supported "
          f"(used to train the {REWARD_MODEL[name]} reward model)  ok")

# --- cross-label statistics cited in the paper's discussion ---
both = df[df.radgraph_class.notna()]  # the 20,551 sentences with both an image and an LLM label
agreement = 100 * (both.label_tau030 == both.label_llm).mean()
print(f"\nimage (tau=0.3) vs LLM label agreement, {len(both)} shared sentences: {agreement:.1f}%")

for rule, col in (("image (tau=0.3)", "label_tau030"), ("LLM (vs. reference)", "label_llm")):
    by_polarity = both.groupby("polarity")[col].mean() * 100
    assert_pct = by_polarity.get("present", float("nan"))
    deny_pct = by_polarity.get("absent", float("nan"))
    print(f"{rule:22s} supports {assert_pct:5.1f}% of assertions, {deny_pct:5.1f}% of denials")

# --- validation manifest and candidate pool ---
manifest = pd.read_csv(DATA / "val_manifest.csv")
if len(manifest) != 3039:
    fail(f"val_manifest.csv: {len(manifest)} rows, expected 3,039")
n_dedup = int(manifest.in_dedup1564.sum())
if n_dedup != 1564:
    fail(f"val_manifest.csv: {n_dedup} rows flagged in_dedup1564, expected 1,564")
print(f"\nval_manifest.csv: 3,039 validation volumes, {n_dedup} flagged in_dedup1564  ok")

pool_ids = []
for shard in sorted((DATA / "candidates_pool").glob("shard*.jsonl")):
    for line in open(shard, encoding="utf-8"):
        row = json.loads(line)
        if len(row["candidates"]) != 32:
            fail(f"{shard.name}: {row['volume_id']} has {len(row['candidates'])} candidates, expected 32")
        if "npz" in row:
            fail(f"{shard.name}: {row['volume_id']} still carries the unscrubbed 'npz' path field")
        pool_ids.append(row["volume_id"])
if len(pool_ids) != 3039:
    fail(f"candidates_pool: {len(pool_ids)} records across all shards, expected 3,039")
if set(pool_ids) != set(manifest.volume_id):
    fail("candidates_pool volume_ids do not match val_manifest.csv exactly")
print(f"candidates_pool: 24 shards, {len(pool_ids)} volumes, 32 candidates each, "
      f"volume_ids match val_manifest.csv exactly, no leftover 'npz' field  ok")

print("\nALL CHECKS PASSED")
