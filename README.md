# gojev

Go harness for [System One](https://docs.typesafe.ai/concepts/system-one) decision models: TypeSafe's hosted **Jev** and the open-weight, self-hostable **Kev**.

Decision models do not generate text.
They take a `state` (a string or structured JSON) plus a map of typed questions and return one calibrated answer per question in a single forward pass.

| Primitive | Ask | Get |
| --- | --- | --- |
| `Bool` | yes/no | probability of yes |
| `Choice` | pick one of up to 255 options | chosen option + probability per option |
| `Score` | rate on an ordered rubric (2-10 levels) | interpolated score + probability per level |

## Where things live

The protocol layer is in the `taigrr` forks of [fantasy](https://github.com/taigrr/fantasy) and [catwalk](https://github.com/taigrr/catwalk):

- `fantasy.EvaluationModel` / `fantasy.EvaluationProvider` — the modality abstraction.
- `fantasy/providers/vercel` — Vercel AI Gateway `POST /v1/evaluate` (model `typesafe-ai/jev`).
- `fantasy/providers/typesafe` — native `POST /v1/systemone` on `api.typesafe.ai`, or any wire-compatible server such as a local Kev.
- `fantasy/providers/kev` — Kev fully in-process (llama.cpp via yzma), weights downloaded at runtime.
- `catwalk.Provider.EvaluationModels` — Jev listed under the `vercel` and `typesafe` providers.

gojev adds what an agent framework needs on top: catwalk-driven construction, typed decision helpers, the Kev conversion tooling, and a CLI.

## Install

```sh
go get github.com/taigrr/gojev
```

## Usage

```go
import (
    "github.com/taigrr/fantasy"
    "github.com/taigrr/fantasy/providers/vercel"
    "github.com/taigrr/gojev"
)

provider, _ := vercel.New(vercel.WithAPIKey(os.Getenv("AI_GATEWAY_API_KEY")))
model, _ := provider.(fantasy.EvaluationProvider).EvaluationModel(ctx, vercel.ModelJev)

resp, err := gojev.Evaluate(ctx, model, "Shoes arrived two weeks late and I see two charges on my card.", gojev.Questions{
    "department": gojev.Choice("Which team should handle this?", map[string]string{
        "returns":  "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing":  "Charges, invoices, payment problems",
    }),
    "escalate":    gojev.Bool("Does this need urgent human attention?"),
    "frustration": gojev.Score("How frustrated is the customer?", "Calm", "Frustrated", "Very angry"),
})

resp.Answers["department"].Choice   // "billing"
resp.Answers["escalate"].Yes()      // true
resp.Answers["frustration"].Level() // "Frustrated"
```

### Typed decisions for gating

```go
type Verdict string
const (Allow Verdict = "allow"; Ask Verdict = "ask"; Deny Verdict = "deny")

d, err := gojev.Classify(ctx, model, toolCall, "Should this run without asking?",
    []Verdict{Allow, Ask, Deny}, map[Verdict]string{Allow: "safe and reversible"})
if v, ok := d.Above(0.8, 0.3); ok && v == Allow { ... } // confident AND unambiguous
```

Also: `gojev.Yes(ctx, model, state, q, threshold)`, `gojev.Ask(...)`, `gojev.Fallback(cloud, localKev)`.

### From catwalk

```go
model, err := gojev.FromCatwalk(ctx, provider, "") // uses provider.DefaultEvaluationModelID
```

Works for catwalk entries of type `vercel` and `typesafe`; `$ENV` API keys are resolved.

### Local Kev, in-process

`fantasy/providers/kev` runs Kev entirely inside your process via llama.cpp (yzma, no cgo).
Nothing is embedded in binaries; weights and the llama.cpp shared libraries are fetched on first use and cached under `$KEV_CACHE` (default: the OS cache dir, `kev/`).

```go
p, _ := kev.New(
    kev.WithCheckpoint(kev.Checkpoint4B), // 0.8B fast on CPU; 4B default; 9B most accurate
    kev.WithAutoLibraries(),              // install pinned, verified llama.cpp if missing
)
defer p.(io.Closer).Close()
model, _ := p.(fantasy.EvaluationProvider).EvaluationModel(ctx, "") // free; loads on first Evaluate
```

#### Trust model

Everything that is downloaded is verified against a digest compiled into the Go module before it is written, loaded or executed:

- **Weights**: `manifestDigests` pins the SHA-256 of each checkpoint's `manifest.json`; every file in the bundle is then verified against that manifest. Corrupt or tampered files are re-fetched; an untrusted manifest is `ErrManifestUntrusted`.
- **llama.cpp**: `LlamaCPPVersion` pins the release *and* the SHA-256 of its digest manifest (the same pin yzma ships). Every archive and every extracted file is verified against it. On every load the installed files are re-hashed; a tampered library is `ErrLibrariesCorrupt` and is never `dlopen`ed.
- **Upgrades**: bumping `LlamaCPPVersion` makes existing installs report `ErrLibrariesOutdated`; with `WithAutoLibraries()` the new release is staged, verified, and swapped in atomically, then the old one removed.

Backend selection is automatic (Metal on Apple Silicon; CUDA → ROCm → Vulkan → CPU elsewhere) and can be forced with `WithProcessor` or `$KEV_PROCESSOR`.

#### Latency (M4 Max, f16, model loaded)

| Checkpoint | 1 q, short state | 3 q, ~100 tok | 8 q, ~100 tok | 1 q, ~1500 tok | 3 q, ~1500 tok |
| --- | --- | --- | --- | --- | --- |
| kev-0.8b (Metal) | 14 ms | 61 ms | 130 ms | 194 ms | 250 ms |
| kev-4b (Metal) | 45 ms | 205 ms | 423 ms | 1.2 s | 1.4 s |
| kev-9b (Metal) | 68 ms | 323 ms | 800 ms | 2.6 s | 2.6 s |
| kev-0.8b (CPU, 12P+4E) | 117 ms | 460 ms | — | 1.2 s | — |
| kev-4b (CPU) | 630 ms | 2.2 s | — | 5.7 s | — |
| kev-9b (CPU) | 1.1 s | — | — | — | — |

Multiple questions share one decode of the state (KV prefix cache), so cost grows with branch tokens, not state × questions.
On CPU-only machines prefer `kev-0.8b`; a Q8_0 4B bundle would roughly halve the 4B CPU numbers and is a `convert.py --outtype q8_0` away.

Output matches Kev's Python server to two decimals on every fixture for all three checkpoints (`tools/kev-convert`).
A running `kev.serve` (Python) is still reachable through `fantasy/providers/typesafe` with `WithBaseURL("http://127.0.0.1:8008")`.

## CLI

```sh
go run ./cmd/gojev -provider vercel \
  -state "rm -rf ./build && git push --force origin main" \
  -q "verdict=choice:Allow without asking?;allow=safe,ask=needs confirmation,deny=never" \
  -q "destructive=bool:Could this cause irreversible data loss?" \
  -q "risk=score:How risky?;harmless,moderate,dangerous"
```

`-provider typesafe|vercel|kev|kev-server`, `-model` picks the Kev checkpoint, `-state -` reads stdin, `-json-state` sends structured state, `-json` prints the raw response, `-zdr` requests zero data retention (Pro/Enterprise gateway plans only).

## Development

```sh
go test ./...
staticcheck ./...
```
