// Command gojev evaluates a state against typed questions using Jev (via
// TypeSafe or Vercel AI Gateway) or a local Kev server.
//
// Questions are given as name=type:instructions[;criteria] flags:
//
//	gojev -state "I was charged twice" \
//	  -q "refund=bool:Is the customer asking for money back?" \
//	  -q "route=choice:Route this ticket;billing=payments,shipping=delivery" \
//	  -q "urgency=score:How urgent?;low,medium,high"
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/typesafe"
	"github.com/taigrr/fantasy/providers/vercel"

	"github.com/taigrr/fantasy/providers/kev"
	"github.com/taigrr/gojev"
	"github.com/taigrr/gojev/internal/version"
)

const (
	providerTypeSafe  = "typesafe"
	providerVercel    = "vercel"
	providerKev       = "kev"
	providerKevServer = "kev-server"

	stateStdin = "-"
)

type questionFlags []string

func (q *questionFlags) String() string     { return strings.Join(*q, " ") }
func (q *questionFlags) Set(v string) error { *q = append(*q, v); return nil }

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "gojev:", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout io.Writer) error {
	flags := flag.NewFlagSet("gojev", flag.ContinueOnError)
	provider := flags.String("provider", providerVercel, "provider: typesafe, vercel, kev (in-process), or kev-server")
	model := flags.String("model", "", "model override (default depends on provider)")
	baseURL := flags.String("base-url", "", "base URL override")
	state := flags.String("state", "", "state text to evaluate, or - to read stdin")
	stateJSON := flags.Bool("json-state", false, "parse state as JSON instead of a string")
	asJSON := flags.Bool("json", false, "print the full response as JSON")
	zdr := flags.Bool("zdr", false, "vercel: require zero data retention")
	timeout := flags.Duration("timeout", 60*time.Second, "request timeout")
	downloadTimeout := flags.Duration("download-timeout", 2*time.Hour, "kev: budget for first-use weight/library download")
	showVersion := flags.Bool("version", false, "print version and exit")
	var questions questionFlags
	flags.Var(&questions, "q", "question as name=type:instructions[;criteria] (repeatable)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *showVersion {
		fmt.Fprintln(stdout, "gojev", version.Version)
		return nil
	}
	if *state == "" {
		return errors.New("-state is required")
	}
	if len(questions) == 0 {
		return errors.New("at least one -q is required")
	}

	stateValue, err := buildState(*state, *stateJSON, stdin)
	if err != nil {
		return err
	}
	parsed, err := parseQuestions(questions)
	if err != nil {
		return err
	}
	evaluator, err := buildModel(context.Background(), *provider, *baseURL, *model, *zdr)
	if err != nil {
		return err
	}
	if loader, ok := evaluator.(interface{ Load(context.Context) error }); ok {
		// First use of a local model may download weights; give that its
		// own generous budget instead of the request timeout.
		loadCtx, cancelLoad := context.WithTimeout(context.Background(), *downloadTimeout)
		err := loader.Load(loadCtx)
		cancelLoad()
		if err != nil {
			return err
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	resp, err := gojev.Evaluate(ctx, evaluator, stateValue, parsed)
	if err != nil {
		return err
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(resp)
	}
	printResponse(stdout, resp)
	return nil
}

func buildState(state string, parseJSON bool, stdin io.Reader) (any, error) {
	if state == stateStdin {
		if stdin == nil {
			return nil, errors.New("no stdin available for -state -")
		}
		data, err := io.ReadAll(stdin)
		if err != nil {
			return nil, fmt.Errorf("read stdin: %w", err)
		}
		state = string(data)
	}
	if !parseJSON {
		return state, nil
	}
	var value any
	if err := json.Unmarshal([]byte(state), &value); err != nil {
		return nil, fmt.Errorf("parse state as JSON: %w", err)
	}
	return value, nil
}

const (
	envTypeSafeAPIKey = "TYPESAFE_API_KEY"
	envGatewayAPIKey  = "AI_GATEWAY_API_KEY"
	envVercelAPIKey   = "VERCEL_API_KEY"
	envOIDCToken      = "VERCEL_OIDC_TOKEN"
)

func firstEnv(names ...string) string {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	return ""
}

func evaluationModel(ctx context.Context, p fantasy.Provider, modelID string) (fantasy.EvaluationModel, error) {
	ep, ok := p.(fantasy.EvaluationProvider)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support evaluation", p.Name())
	}
	return ep.EvaluationModel(ctx, modelID)
}

func buildModel(ctx context.Context, provider, baseURL, model string, zdr bool) (fantasy.EvaluationModel, error) {
	switch provider {
	case providerTypeSafe:
		opts := []typesafe.Option{typesafe.WithAPIKey(firstEnv(envTypeSafeAPIKey))}
		if baseURL != "" {
			opts = append(opts, typesafe.WithBaseURL(baseURL))
		}
		p, err := typesafe.New(opts...)
		if err != nil {
			return nil, err
		}
		return evaluationModel(ctx, p, model)
	case providerVercel:
		opts := []vercel.Option{vercel.WithAPIKey(firstEnv(envGatewayAPIKey, envOIDCToken, envVercelAPIKey))}
		if baseURL != "" {
			opts = append(opts, vercel.WithBaseURL(baseURL))
		}
		if zdr {
			opts = append(opts, vercel.WithEvaluationOptions(vercel.EvaluationOptions{ZeroDataRetention: true}))
		}
		p, err := vercel.New(opts...)
		if err != nil {
			return nil, err
		}
		return evaluationModel(ctx, p, model)
	case providerKev:
		// In-process Kev via llama.cpp. -model selects the checkpoint
		// (kev-0.8b, kev-4b, kev-9b, or a local bundle dir); weights and the
		// llama.cpp libraries are downloaded on first use.
		opts := []kev.Option{kev.WithAutoLibraries()}
		if model != "" && model != kev.ModelLatest {
			opts = append(opts, kev.WithCheckpoint(kev.Checkpoint(model)))
		}
		opts = append(opts, kev.WithDownloader(kev.Downloader{Progress: func(file string, done, total int64) {
			if total > 0 {
				fmt.Fprintf(os.Stderr, "\r%s %3d%%", file, done*100/total)
				if done == total {
					fmt.Fprintln(os.Stderr)
				}
			}
		}}))
		p, err := kev.New(opts...)
		if err != nil {
			return nil, err
		}
		return evaluationModel(ctx, p, "")
	case providerKevServer:
		// A running kev.serve (Python) or any TypeSafe-compatible server.
		opts := []typesafe.Option{typesafe.WithName("kev"), typesafe.WithAPIKey("local")}
		if baseURL == "" {
			baseURL = "http://127.0.0.1:8008"
		}
		opts = append(opts, typesafe.WithBaseURL(baseURL))
		p, err := typesafe.New(opts...)
		if err != nil {
			return nil, err
		}
		if model == "" {
			model = typesafe.ModelKevLatest
		}
		return evaluationModel(ctx, p, model)
	default:
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
}

func parseQuestions(specs []string) (gojev.Questions, error) {
	out := make(gojev.Questions, len(specs))
	for _, spec := range specs {
		name, question, err := parseQuestion(spec)
		if err != nil {
			return nil, fmt.Errorf("-q %q: %w", spec, err)
		}
		if _, dup := out[name]; dup {
			return nil, fmt.Errorf("-q %q: duplicate question name %q", spec, name)
		}
		out[name] = question
	}
	return out, nil
}

// splitList splits a comma-separated list, trimming whitespace and dropping
// empty entries. Commas cannot be escaped; descriptions and level names
// containing commas are not expressible on the command line.
func splitList(list string) []string {
	var parts []string
	for part := range strings.SplitSeq(list, ",") {
		if part = strings.TrimSpace(part); part != "" {
			parts = append(parts, part)
		}
	}
	return parts
}

func parseQuestion(spec string) (string, fantasy.EvaluationQuestion, error) {
	name, rest, ok := strings.Cut(spec, "=")
	name = strings.TrimSpace(name)
	if !ok || name == "" {
		return "", fantasy.EvaluationQuestion{}, errors.New("expected name=type:instructions")
	}
	kind, rest, ok := strings.Cut(rest, ":")
	if !ok {
		return "", fantasy.EvaluationQuestion{}, errors.New("expected type:instructions")
	}
	instructions, criteria, _ := strings.Cut(rest, ";")
	instructions = strings.TrimSpace(instructions)
	if instructions == "" {
		return "", fantasy.EvaluationQuestion{}, errors.New("instructions are empty")
	}
	switch strings.TrimSpace(kind) {
	case "bool", "boolean", "noul":
		trueCriteria, falseCriteria := "", ""
		if criteria != "" {
			trueCriteria, falseCriteria, _ = strings.Cut(criteria, ",")
			trueCriteria, falseCriteria = strings.TrimSpace(trueCriteria), strings.TrimSpace(falseCriteria)
		}
		return name, fantasy.BoolQuestionWithCriteria(instructions, trueCriteria, falseCriteria), nil
	case "choice":
		options := map[string]string{}
		for _, part := range splitList(criteria) {
			option, description, _ := strings.Cut(part, "=")
			option = strings.TrimSpace(option)
			if option == "" {
				return "", fantasy.EvaluationQuestion{}, fmt.Errorf("choice option %q has an empty name", part)
			}
			if _, dup := options[option]; dup {
				return "", fantasy.EvaluationQuestion{}, fmt.Errorf("duplicate choice option %q", option)
			}
			options[option] = strings.TrimSpace(description)
		}
		if len(options) == 0 {
			return "", fantasy.EvaluationQuestion{}, errors.New("choice needs at least one option after ';'")
		}
		return name, gojev.Choice(instructions, options), nil
	case "score":
		levels := splitList(criteria)
		if len(levels) < 2 {
			return "", fantasy.EvaluationQuestion{}, errors.New("score needs at least two levels after ';'")
		}
		seen := make(map[string]struct{}, len(levels))
		for _, level := range levels {
			if _, dup := seen[level]; dup {
				return "", fantasy.EvaluationQuestion{}, fmt.Errorf("duplicate score level %q", level)
			}
			seen[level] = struct{}{}
		}
		return name, gojev.Score(instructions, levels...), nil
	default:
		return "", fantasy.EvaluationQuestion{}, fmt.Errorf("unknown type %q", kind)
	}
}

func printResponse(w io.Writer, resp *fantasy.EvaluationResponse) {
	names := make([]string, 0, len(resp.Answers))
	for name := range resp.Answers {
		names = append(names, name)
	}
	sort.Strings(names)
	fmt.Fprintf(w, "model: %s\n", resp.Model)
	for _, name := range names {
		answer := resp.Answers[name]
		switch answer.Type {
		case fantasy.EvaluationQuestionTypeBool:
			fmt.Fprintf(w, "%s: %t (p=%.3f)\n", name, answer.Yes(), answer.Probability)
		case fantasy.EvaluationQuestionTypeChoice:
			fmt.Fprintf(w, "%s: %s", name, answer.Choice)
			if answer.Confidence != nil {
				fmt.Fprintf(w, " (confidence=%.3f)", *answer.Confidence)
			}
			fmt.Fprintln(w)
			printProbabilities(w, answer.Probabilities)
		case fantasy.EvaluationQuestionTypeScore:
			fmt.Fprintf(w, "%s: %.2f %s", name, answer.Score, answer.Level())
			if answer.Confidence != nil {
				fmt.Fprintf(w, " (confidence=%.3f)", *answer.Confidence)
			}
			fmt.Fprintln(w)
			for i, prob := range answer.LevelProbabilities {
				label := ""
				if i < len(answer.Levels) {
					label = answer.Levels[i]
				}
				fmt.Fprintf(w, "  %d %s: %.3f\n", i, label, prob)
			}
		default:
			fmt.Fprintf(w, "%s: (unrecognised answer type %q)\n", name, answer.Type)
		}
	}
	fmt.Fprintf(w, "usage: in=%d out=%d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

func printProbabilities(w io.Writer, probs map[string]float64) {
	keys := make([]string, 0, len(probs))
	for key := range probs {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if probs[keys[i]] != probs[keys[j]] {
			return probs[keys[i]] > probs[keys[j]]
		}
		return keys[i] < keys[j]
	})
	for _, key := range keys {
		fmt.Fprintf(w, "  %s: %.3f\n", key, probs[key])
	}
}
