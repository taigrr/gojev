// Command gojev evaluates a state against typed questions using Jev (via
// TypeSafe or Vercel AI Gateway), Clef (via Cloudflare Workers AI) or a
// local Kev.
//
// Questions are given as name=type:instructions[;criteria] flags:
//
//	gojev -state "I was charged twice" \
//	  -q "refund=bool:Is the customer asking for money back?" \
//	  -q "route=choice:Route this ticket;billing=payments,shipping=delivery" \
//	  -q "urgency=score:How urgent?;low,medium,high"
//
// Programs driving gojev (such as the npm wrapper) pass a whole request as
// JSON instead, in the System One shape, and read the response with -json:
//
//	echo '{"state":"I was charged twice","questions":{"refund":{"type":"noul","instructions":"Refund?"}}}' \
//	  | gojev -input - -json
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
	"github.com/taigrr/fantasy/providers/cloudflare"
	"github.com/taigrr/fantasy/providers/kev"
	"github.com/taigrr/fantasy/providers/typesafe"
	"github.com/taigrr/fantasy/providers/vercel"
	"github.com/taigrr/gojev"
	"github.com/taigrr/gojev/internal/version"
)

const (
	providerTypeSafe   = "typesafe"
	providerVercel     = "vercel"
	providerCloudflare = "cloudflare"
	providerKev        = "kev"
	providerKevServer  = "kev-server"

	stateStdin = "-"

	defaultKevServerURL = "http://127.0.0.1:8008"
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
	provider := flags.String("provider", providerVercel, "provider: typesafe, vercel, cloudflare, kev (in-process), or kev-server")
	model := flags.String("model", "", "model override (default depends on provider)")
	baseURL := flags.String("base-url", "", "base URL override (cloudflare: the Workers AI run root, e.g. an AI Gateway)")
	accountID := flags.String("account", "", "cloudflare: account id (default $"+envCloudflareAccountID+")")
	gatewayID := flags.String("gateway", "", "cloudflare: route through the named AI Gateway")
	state := flags.String("state", "", "state text to evaluate, or - to read stdin")
	stateJSON := flags.Bool("json-state", false, "parse state as JSON instead of a string")
	input := flags.String("input", "", "read a JSON {state, questions} request from this file, or - for stdin (replaces -state/-q)")
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
	stateValue, parsed, err := buildRequest(*input, *state, *stateJSON, questions, stdin)
	if err != nil {
		return err
	}
	evaluator, err := buildModel(context.Background(), modelConfig{
		provider:  *provider,
		baseURL:   *baseURL,
		model:     *model,
		accountID: *accountID,
		gatewayID: *gatewayID,
		zdr:       *zdr,
	})
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

// buildRequest resolves the state and questions from either -input or the
// -state/-q flags; mixing the two is an error.
func buildRequest(input, state string, parseJSON bool, questions questionFlags, stdin io.Reader) (any, gojev.Questions, error) {
	if input != "" {
		if state != "" || len(questions) > 0 {
			return nil, nil, errors.New("-input cannot be combined with -state or -q")
		}
		return readInput(input, stdin)
	}
	if state == "" {
		return nil, nil, errors.New("-state or -input is required")
	}
	if len(questions) == 0 {
		return nil, nil, errors.New("at least one -q is required")
	}
	stateValue, err := buildState(state, parseJSON, stdin)
	if err != nil {
		return nil, nil, err
	}
	parsed, err := parseQuestions(questions)
	if err != nil {
		return nil, nil, err
	}
	return stateValue, parsed, nil
}

// inputRequest is the JSON request accepted by -input. Questions use the
// System One wire shape (type noul|choice|score with criteria); fantasy's
// own shape (type bool with options/levels/true_criteria/false_criteria) is
// accepted too.
type inputRequest struct {
	State     json.RawMessage          `json:"state"`
	Questions map[string]inputQuestion `json:"questions"`
}

type inputQuestion struct {
	Type          string            `json:"type"`
	Instructions  string            `json:"instructions"`
	Criteria      json.RawMessage   `json:"criteria,omitempty"`
	Options       map[string]string `json:"options,omitempty"`
	Levels        []string          `json:"levels,omitempty"`
	TrueCriteria  string            `json:"true_criteria,omitempty"`
	FalseCriteria string            `json:"false_criteria,omitempty"`
}

func readInput(source string, stdin io.Reader) (any, gojev.Questions, error) {
	var data []byte
	var err error
	if source == stateStdin {
		if stdin == nil {
			return nil, nil, errors.New("no stdin available for -input -")
		}
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(source)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("read -input: %w", err)
	}
	var req inputRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, nil, fmt.Errorf("parse -input: %w", err)
	}
	if len(req.State) == 0 || string(req.State) == "null" {
		return nil, nil, errors.New("-input: state is required")
	}
	if len(req.Questions) == 0 {
		return nil, nil, errors.New("-input: at least one question is required")
	}
	var state any
	if err := json.Unmarshal(req.State, &state); err != nil {
		return nil, nil, fmt.Errorf("-input: state: %w", err)
	}
	questions := make(gojev.Questions, len(req.Questions))
	for name, question := range req.Questions {
		if strings.TrimSpace(name) == "" {
			return nil, nil, errors.New("-input: question name is required")
		}
		parsed, err := decodeInputQuestion(question)
		if err != nil {
			return nil, nil, fmt.Errorf("-input: question %q: %w", name, err)
		}
		questions[name] = parsed
	}
	return state, questions, nil
}

func decodeInputQuestion(in inputQuestion) (fantasy.EvaluationQuestion, error) {
	hasCriteria := len(in.Criteria) > 0 && string(in.Criteria) != "null"
	switch strings.TrimSpace(in.Type) {
	case "bool", "boolean", typesafe.WireTypeNoul:
		trueCriteria, falseCriteria := in.TrueCriteria, in.FalseCriteria
		if hasCriteria {
			var criteria map[string]string
			if err := json.Unmarshal(in.Criteria, &criteria); err != nil {
				return fantasy.EvaluationQuestion{}, fmt.Errorf("criteria must be a {true, false} object: %w", err)
			}
			trueCriteria, falseCriteria = criteria["true"], criteria["false"]
		}
		return fantasy.BoolQuestionWithCriteria(in.Instructions, trueCriteria, falseCriteria), nil
	case typesafe.WireTypeChoice:
		options := in.Options
		if hasCriteria {
			var criteria map[string]*string
			if err := json.Unmarshal(in.Criteria, &criteria); err != nil {
				return fantasy.EvaluationQuestion{}, fmt.Errorf("criteria must be an option-to-description object: %w", err)
			}
			options = make(map[string]string, len(criteria))
			for option, description := range criteria {
				if description != nil {
					options[option] = *description
					continue
				}
				options[option] = ""
			}
		}
		if len(options) == 0 {
			return fantasy.EvaluationQuestion{}, errors.New("choice needs at least one option")
		}
		return gojev.Choice(in.Instructions, options), nil
	case typesafe.WireTypeScore:
		levels := in.Levels
		if hasCriteria {
			if err := json.Unmarshal(in.Criteria, &levels); err != nil {
				return fantasy.EvaluationQuestion{}, fmt.Errorf("criteria must be an array of level names: %w", err)
			}
		}
		if len(levels) < 2 {
			return fantasy.EvaluationQuestion{}, errors.New("score needs at least two levels")
		}
		return gojev.Score(in.Instructions, levels...), nil
	default:
		return fantasy.EvaluationQuestion{}, fmt.Errorf("unknown type %q", in.Type)
	}
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
	envTypeSafeAPIKey       = "TYPESAFE_API_KEY"
	envGatewayAPIKey        = "AI_GATEWAY_API_KEY"
	envVercelAPIKey         = "VERCEL_API_KEY"
	envOIDCToken            = "VERCEL_OIDC_TOKEN"
	envCloudflareAPIToken   = "CLOUDFLARE_API_TOKEN"
	envCloudflareAuthToken  = "CLOUDFLARE_AUTH_TOKEN"
	envCloudflareAccountID  = gojev.EnvCloudflareAccountID
	envCloudflareGatewayID  = "CLOUDFLARE_GATEWAY_ID"
	envCloudflareWorkersURL = "CLOUDFLARE_WORKERS_AI_URL"
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

// modelConfig carries the CLI's provider selection.
type modelConfig struct {
	provider  string
	baseURL   string
	model     string
	accountID string
	gatewayID string
	zdr       bool
}

func buildModel(ctx context.Context, cfg modelConfig) (fantasy.EvaluationModel, error) {
	baseURL, model := cfg.baseURL, cfg.model
	switch cfg.provider {
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
		if cfg.zdr {
			opts = append(opts, vercel.WithEvaluationOptions(vercel.EvaluationOptions{ZeroDataRetention: true}))
		}
		p, err := vercel.New(opts...)
		if err != nil {
			return nil, err
		}
		return evaluationModel(ctx, p, model)
	case providerCloudflare:
		// Clef on Workers AI. -base-url replaces the run root entirely (for
		// example an AI Gateway); otherwise it is derived from the account.
		opts := []cloudflare.Option{cloudflare.WithAPIKey(firstEnv(envCloudflareAPIToken, envCloudflareAuthToken))}
		if baseURL == "" {
			baseURL = firstEnv(envCloudflareWorkersURL)
		}
		if baseURL != "" {
			opts = append(opts, cloudflare.WithBaseURL(baseURL))
		} else {
			accountID := cfg.accountID
			if accountID == "" {
				accountID = firstEnv(envCloudflareAccountID)
			}
			if accountID == "" {
				return nil, fmt.Errorf("cloudflare: -account, $%s or -base-url is required", envCloudflareAccountID)
			}
			opts = append(opts, cloudflare.WithAccountID(accountID))
		}
		if gatewayID := firstNonEmpty(cfg.gatewayID, firstEnv(envCloudflareGatewayID)); gatewayID != "" {
			opts = append(opts, cloudflare.WithGatewayID(gatewayID))
		}
		p, err := cloudflare.New(opts...)
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
			baseURL = defaultKevServerURL
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
		return nil, fmt.Errorf("unknown provider %q", cfg.provider)
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
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
