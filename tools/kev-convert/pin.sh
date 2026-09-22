#!/usr/bin/env bash
# Print the Go map literal for fantasy/providers/kev manifestDigests from converted bundles.
# Usage: ./pin.sh out/kev-0.8b out/kev-4b out/kev-9b
set -euo pipefail
echo "var manifestDigests = map[Checkpoint]string{"
for dir in "$@"; do
  name="$(basename "$dir")"
  const="Checkpoint$(echo "${name#kev-}" | tr 'a-z.' 'A-Z_')"
  printf '\t%s: "%s",\n' "$const" "$(shasum -a 256 "$dir/manifest.json" | cut -d' ' -f1)"
done
echo "}"
