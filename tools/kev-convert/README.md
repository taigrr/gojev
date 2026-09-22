# kev-convert

Turns a [Kev](https://github.com/jaredpalmer/kev) checkpoint (PEFT LoRA adapter + `head.pt`) into a
self-contained GGUF bundle that `github.com/taigrr/fantasy/providers/kev` can load in-process.

Requires `uv` and a [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout for `convert_hf_to_gguf.py`.

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp
uv run convert.py --run jaredpalmer/kev-4b --out out/kev-4b --llama-cpp ./llama.cpp
uv run reference.py --run jaredpalmer/kev-4b --out out/kev-4b/reference.json   # optional: parity fixtures
./publish.sh taigrr out/kev-4b                                                   # needs `hf auth login`
```

`convert.py` merges the adapter into the base in fp32 exactly as `kev/checkpoint.py` does with `KEV_MERGE=1`,
exports with `--no-mtp` (the Go path never generates), and writes:

| File | Contents |
| --- | --- |
| `model-f16.gguf` | merged backbone |
| `head.json` | pointer head `q`/`k` weights, calibration temperature, dims |
| `manifest.json` | sha256 + size of each file; the Go downloader verifies these |

Parity: with `KEV_TEST_BUNDLE=out/kev-4b go test -run Integration ./providers/kev` in fantasy, Go output
matches the Python reference to two decimals on all fixtures for 0.8B, 4B and 9B.

After converting, regenerate the manifest pins for `fantasy/providers/kev/download.go`:

```sh
./pin.sh out/kev-0.8b out/kev-4b out/kev-9b
```

Bumping llama.cpp: set `LlamaCPPVersion` in `fantasy/providers/kev/libs.go` to the `<tag>@sha256:<manifest digest>`
value printed in the corresponding [llama-cpp-builder](https://github.com/hybridgroup/llama-cpp-builder/releases)
release notes (it must match the yzma version in `go.mod`), then re-run the parity tests.
