# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers>=5.17", "peft", "safetensors", "huggingface_hub", "numpy", "pydantic"]
# ///
"""Produce reference outputs from the Python Kev model for parity testing.

Writes a JSON list of {state, questions, answers} using kev.api exactly as the server does.
Run from the kev-convert dir: uv run reference.py --run jaredpalmer/kev-0.8b --out out/kev-0.8b/reference.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "kev"))

CASES = [
    {
        "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. Fix this today.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                        "shipping": "Delivery status, delays, lost packages",
                                        "billing": "Charges, invoices, payment problems"}},
            "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Frustrated", "Very angry"]},
        },
    },
    {
        "state": "The agent wants to run: rm -rf ./build && git push --force origin main",
        "questions": {
            "verdict": {"type": "choice", "instructions": "Should this command be allowed to run without asking?",
                        "criteria": {"allow": "safe and reversible", "ask": "needs human confirmation",
                                     "deny": "destructive and should never run"}},
            "destructive": {"type": "noul", "instructions": "Could this command cause irreversible data loss?",
                            "criteria": {"true": "deletes or overwrites data with no recovery", "false": "read-only or trivially undone"}},
        },
    },
    {
        "state": {"messages": [{"role": "user", "content": "hi, can you change my email address?"}]},
        "questions": {
            "intent": {"type": "choice", "instructions": "What does the user want?",
                       "criteria": {"account_update": None, "billing": None, "support": None, "none": None}},
        },
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.api import SystemOneRequest, to_record, to_answers
    from kev.model import encode

    ck = Checkpoint(args.run)
    tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32))
    print(f"temperature={model.head.temperature}")

    results = []
    for case in CASES:
        # sort choice criteria so option order matches the Go encoder
        qs = {}
        for qid, q in case["questions"].items():
            q = dict(q)
            if q["type"] == "choice":
                q["criteria"] = {k: q["criteria"][k] for k in sorted(q["criteria"])}
            qs[qid] = q
        req = SystemOneRequest.model_validate({"state": case["state"], "model": "kev-latest", "questions": qs})
        rec, meta = to_record(req)
        enc = encode(tok, rec, max_state=8192, max_branch=8192)
        probs = [p.tolist() for p in model.probs(enc)]
        answers = to_answers(probs, meta)
        results.append({"state": case["state"], "questions": qs, "answers": answers, "raw_probs": probs,
                        "n_tokens": len(enc["ids"])})
        print(json.dumps(answers))

    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
