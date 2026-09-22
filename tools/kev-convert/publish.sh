#!/usr/bin/env bash
# Publish converted Kev GGUF bundles to Hugging Face under <owner>/<name>-gguf.
# Usage: ./publish.sh taigrr out/kev-0.8b [out/kev-4b ...]
# Requires: hf auth login (write token).
set -euo pipefail
owner="$1"; shift
for dir in "$@"; do
  name="$(basename "$dir")"
  repo="$owner/$name-gguf"
  echo "== $repo"
  hf repo create "$repo" --repo-type model --exist-ok >/dev/null
  cat > "$dir/README.md" <<EOF
---
license: apache-2.0
base_model: $(python3 -c "import json;print(json.load(open('$dir/manifest.json'))['base'])")
tags: [kev, jev, decision-model, gguf, evaluation]
---
# $name (GGUF)

[Kev](https://github.com/jaredpalmer/kev) checkpoint \`jaredpalmer/$name\` with its LoRA adapter merged into the base
weights (fp32 merge, stored f16) plus the pointer head in \`head.json\`. Produced by
[gojev/kev-convert](https://github.com/taigrr/gojev) for in-process use from Go via
\`github.com/taigrr/fantasy/providers/kev\`.

Files are checksummed in \`manifest.json\`; the Go loader verifies them on download.
EOF
  hf upload "$repo" "$dir" . --commit-message "publish $name bundle"
done
