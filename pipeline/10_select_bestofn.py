"""Best-of-N selection: score N sampled candidates with a trained PRM and
return the one with the highest mean sentence-support probability.

Implements Eq. 2 of the paper directly, with no dependence on TRL's internal
tokenisation of the training format: for a candidate split into sentences
s_1..s_m, the classifier is read out at the final token of the text
"PROMPT \\n s_1 \\n ... \\n s_j" for each j, giving p_j = P(supported | ...);
the candidate score is S(c) = mean(p_1..p_m), and the returned report is the
candidate with the highest S(c). Because attention is causal, this reproduces
the same per-sentence probabilities as a single forward pass over the whole
candidate, just less efficiently -- readability over speed for a reference
script. step_separator matches the "\\n" used in 09_train_prm.py.

Usage:  10_select_bestofn.py --model work/prm_0.5b/final --candidates <path>.jsonl --out selected.jsonl

Input format (one line per volume):
    {"volume_id": "...", "candidates": ["report text 1", "report text 2", ...]}
Output format (one line per volume):
    {"volume_id": "...", "selected_index": i, "scores": [S(c_0), S(c_1), ...]}
"""
import argparse
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import PROMPT, split_report

STEP_SEPARATOR = "\n"


@torch.no_grad()
def score_candidate(report: str, tokenizer, model, device) -> float:
    """S(c) of Equation 2: mean sentence-support probability."""
    sentences = split_report(report)
    if not sentences:
        return 0.0
    probs, text = [], PROMPT
    for sentence in sentences:
        text = f"{text}{STEP_SEPARATOR}{sentence}"
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        logits = model(**ids).logits[0, -1]  # read-out at the final token
        probs.append(torch.softmax(logits, dim=-1)[1].item())  # P(supported)
    return sum(probs) / len(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained PRM directory (09_train_prm.py --out .../final)")
    ap.add_argument("--candidates", required=True, help="jsonl, one {volume_id, candidates: [...]} per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(device).eval()

    n_volumes = 0
    with open(a.candidates, encoding="utf-8") as f_in, open(a.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            scores = [score_candidate(c, tokenizer, model, device) for c in row["candidates"]]
            selected = max(range(len(scores)), key=scores.__getitem__)
            f_out.write(json.dumps({
                "volume_id": row["volume_id"],
                "selected_index": selected,
                "scores": scores,
            }, ensure_ascii=False) + "\n")
            n_volumes += 1
    print(f"selected 1 of N candidates for {n_volumes} volumes -> {a.out}")


if __name__ == "__main__":
    main()
