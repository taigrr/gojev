# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers>=5.17",
#   "peft",
#   "safetensors",
#   "huggingface_hub",
#   "numpy",
#   "sentencepiece",
#   "gguf",
# ]
# ///
"""Convert a Kev checkpoint (LoRA adapter + head.pt) into a self-contained GGUF bundle.

Output directory contains:
  model-f16.gguf   base weights with the LoRA adapter merged in fp32, stored as f16
  head.json        pointer head weights (q/k linear), temperature, dims, base model id
  manifest.json    sha256 + sizes of every file, for the Go downloader

Usage:
  uv run convert.py --run jaredpalmer/kev-0.8b --out out/kev-0.8b --llama-cpp ../llama.cpp
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Hub id like jaredpalmer/kev-0.8b (optionally @revision) or local dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama-cpp", required=True, help="path to a llama.cpp checkout (for convert_hf_to_gguf.py)")
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32", "q8_0"])
    args = ap.parse_args()

    repo, _, revision = args.run.partition("@")
    run = args.run if os.path.isdir(args.run) else snapshot_download(
        repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]
    )
    run = Path(run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = torch.load(run / "head.pt", map_location="cpu")
    base = meta["base"]
    base_revision = meta.get("base_revision")
    if meta.get("special_embeddings"):
        sys.exit("checkpoint carries trained special embeddings; merging path not supported yet")
    if meta.get("option_isolation"):
        sys.exit("checkpoint uses option isolation; unsupported")

    print(f"base={base} rev={base_revision} lora={meta.get('lora')} temperature={meta.get('temperature')}")

    # Merge LoRA into the base in fp32, exactly as kev/checkpoint.py does when KEV_MERGE=1.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        merged_dir = Path(tmp) / "merged"
        print("loading base in fp32 ...")
        # kev wraps AutoModelForCausalLM(...).model; do the same so the adapter's module names line up,
        # then merge and write the whole causal model back out for the GGUF converter.
        causal = AutoModelForCausalLM.from_pretrained(base, revision=base_revision, dtype=torch.float32)
        print("applying adapter ...")
        peft_backbone = PeftModel.from_pretrained(causal.model, str(run))
        causal.model = peft_backbone.merge_and_unload()
        causal.save_pretrained(merged_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(base, revision=base_revision).save_pretrained(merged_dir)

        gguf_path = out / f"model-{args.outtype}.gguf"
        print(f"converting to {gguf_path} ...")
        subprocess.check_call(
            [sys.executable, str(Path(args.llama_cpp) / "convert_hf_to_gguf.py"), str(merged_dir),
             "--outfile", str(gguf_path), "--outtype", args.outtype, "--no-mtp"]
        )

    head = meta["head"]
    head_json = {
        "format": 1,
        "base": base,
        "base_revision": base_revision,
        "run": args.run,
        "hidden_size": int(head["q.weight"].shape[1]),
        "head_dim": int(head["q.weight"].shape[0]),
        "temperature": float(meta.get("temperature", 1.0)),
        "q_weight": head["q.weight"].float().tolist(),
        "q_bias": head["q.bias"].float().tolist(),
        "k_weight": head["k.weight"].float().tolist(),
        "k_bias": head["k.bias"].float().tolist(),
    }
    (out / "head.json").write_text(json.dumps(head_json))

    files = {}
    for p in sorted(out.iterdir()):
        if p.name in ("manifest.json", "README.md") or p.name.startswith("reference"):
            continue
        files[p.name] = {"sha256": sha256(p), "size": p.stat().st_size}
    manifest = {
        "format": 1,
        "run": args.run,
        "base": base,
        "temperature": head_json["temperature"],
        "hidden_size": head_json["hidden_size"],
        "head_dim": head_json["head_dim"],
        "model": gguf_path.name,
        "head": "head.json",
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
