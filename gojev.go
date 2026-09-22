// Package gojev is a Go harness for "System One" decision models such as
// TypeSafe's hosted Jev and the open-weight, self-hostable Kev.
//
// The wire protocols and the EvaluationModel abstraction live in
// github.com/taigrr/fantasy (providers/typesafe and providers/vercel). This
// package adds what an agent framework needs on top:
//
//   - construction of a fantasy.EvaluationModel from a catwalk provider entry
//   - decision helpers: thresholds, margins, typed Classify, Fallback chains
//   - a launcher for a local Kev server (providers/kev)
//   - a CLI (cmd/gojev)
package gojev

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/taigrr/catwalk/pkg/catwalk"
	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/kev"
	"github.com/taigrr/fantasy/providers/typesafe"
	"github.com/taigrr/fantasy/providers/vercel"
)

// Bool builds a yes/no question. See fantasy.BoolQuestionWithCriteria to
// describe what counts as yes and no.
func Bool(instructions string) fantasy.EvaluationQuestion {
	return fantasy.BoolQuestion(instructions)
}

// Choice builds a pick-one question over named options with optional
// descriptions.
func Choice(instructions string, options map[string]string) fantasy.EvaluationQuestion {
	return fantasy.ChoiceQuestion(instructions, options)
}

// Score builds an ordered-rubric question, levels lowest to highest.
func Score(instructions string, levels ...string) fantasy.EvaluationQuestion {
	return fantasy.ScoreQuestion(instructions, levels...)
}

// Questions is the named set of questions evaluated in one call.
type Questions = map[string]fantasy.EvaluationQuestion

// ErrNoEvaluators is returned by a Fallback with no working members.
var ErrNoEvaluators = errors.New("gojev: no evaluation models available")

// ErrUnsupportedProvider is returned when a catwalk provider has no
// evaluation-capable fantasy implementation.
var ErrUnsupportedProvider = errors.New("gojev: provider does not support evaluation models")

// Evaluate is a convenience wrapper around model.Evaluate.
func Evaluate(ctx context.Context, model fantasy.EvaluationModel, state any, questions Questions) (*fantasy.EvaluationResponse, error) {
	return model.Evaluate(ctx, fantasy.EvaluationCall{State: state, Questions: questions})
}

// Ask evaluates a single question and returns its answer.
func Ask(ctx context.Context, model fantasy.EvaluationModel, state any, question fantasy.EvaluationQuestion) (fantasy.EvaluationAnswer, error) {
	const key = "q"
	resp, err := Evaluate(ctx, model, state, Questions{key: question})
	if err != nil {
		return fantasy.EvaluationAnswer{}, err
	}
	answer, ok := resp.Answers[key]
	if !ok {
		return fantasy.EvaluationAnswer{}, errors.New("gojev: response missing answer")
	}
	return answer, nil
}

// Decision is a typed Choice result.
type Decision[T ~string] struct {
	Selected      T
	Probabilities map[T]float64
	Confidence    *float64
}

// Margin is the gap between the top two option probabilities.
func (d Decision[T]) Margin() float64 {
	first, second := 0.0, 0.0
	for _, p := range d.Probabilities {
		if p > first {
			first, second = p, first
		} else if p > second {
			second = p
		}
	}
	if len(d.Probabilities) == 1 {
		return first
	}
	return first - second
}

// Above returns the selected option only if its probability is at least
// threshold and the margin over the runner-up is at least minMargin.
func (d Decision[T]) Above(threshold, minMargin float64) (T, bool) {
	var zero T
	if d.Probabilities[d.Selected] < threshold || d.Margin() < minMargin {
		return zero, false
	}
	return d.Selected, true
}

// Classify asks a Choice question over the given typed options and returns a
// typed Decision. Descriptions may be nil or omit options.
func Classify[T ~string](ctx context.Context, model fantasy.EvaluationModel, state any, instructions string, options []T, descriptions map[T]string) (Decision[T], error) {
	wire := make(map[string]string, len(options))
	for _, option := range options {
		wire[string(option)] = descriptions[option]
	}
	answer, err := Ask(ctx, model, state, Choice(instructions, wire))
	if err != nil {
		return Decision[T]{}, err
	}
	decision := Decision[T]{
		Selected:      T(answer.Choice),
		Probabilities: make(map[T]float64, len(answer.Probabilities)),
		Confidence:    answer.Confidence,
	}
	for option, p := range answer.Probabilities {
		decision.Probabilities[T(option)] = p
	}
	return decision, nil
}

// Probability is a typed Bool result.
type Probability float64

// Above reports whether the probability meets threshold.
func (p Probability) Above(threshold float64) bool { return float64(p) >= threshold }

// Yes reports whether the probability is at least 0.5.
func (p Probability) Yes() bool { return p.Above(0.5) }

// AskBool asks a yes/no question and returns P(yes).
func AskBool(ctx context.Context, model fantasy.EvaluationModel, state any, instructions string) (Probability, error) {
	answer, err := Ask(ctx, model, state, Bool(instructions))
	if err != nil {
		return 0, err
	}
	return Probability(answer.Probability), nil
}

// Rating is a typed Score result.
type Rating struct {
	// Score is the expected level index, 0 to len(Levels)-1.
	Score float64
	// Levels is the rubric, lowest to highest.
	Levels []string
	// Probabilities holds the probability of each level.
	Probabilities []float64
	Confidence    *float64
}

// Level returns the label nearest the expected score.
func (r Rating) Level() string {
	if len(r.Levels) == 0 {
		return ""
	}
	idx := min(max(int(r.Score+0.5), 0), len(r.Levels)-1)
	return r.Levels[idx]
}

// AtLeast reports whether the expected score reaches the named level.
func (r Rating) AtLeast(level string) bool {
	for i, name := range r.Levels {
		if name == level {
			return r.Score >= float64(i)
		}
	}
	return false
}

// Rate asks a Score question and returns a typed Rating.
func Rate(ctx context.Context, model fantasy.EvaluationModel, state any, instructions string, levels ...string) (Rating, error) {
	answer, err := Ask(ctx, model, state, Score(instructions, levels...))
	if err != nil {
		return Rating{}, err
	}
	rating := Rating{Score: answer.Score, Levels: answer.Levels, Probabilities: answer.LevelProbabilities, Confidence: answer.Confidence}
	if len(rating.Levels) == 0 {
		rating.Levels = levels
	}
	return rating, nil
}

// Fallback chains models: each call tries them in order and returns the first
// success. Errors from all members are joined when every member fails.
func Fallback(models ...fantasy.EvaluationModel) fantasy.EvaluationModel {
	return &fallback{models: models}
}

type fallback struct {
	models []fantasy.EvaluationModel
}

func (f *fallback) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if len(f.models) == 0 {
		return nil, ErrNoEvaluators
	}
	var errs []error
	for _, model := range f.models {
		resp, err := model.Evaluate(ctx, call)
		if err == nil {
			return resp, nil
		}
		if ctx.Err() != nil {
			return nil, err
		}
		errs = append(errs, fmt.Errorf("%s/%s: %w", model.Provider(), model.Model(), err))
	}
	return nil, errors.Join(errs...)
}

func (f *fallback) Provider() string {
	names := make([]string, 0, len(f.models))
	for _, model := range f.models {
		names = append(names, model.Provider())
	}
	return strings.Join(names, ",")
}

// Model returns the first member's model id. Use EvaluationResponse.Model to
// learn which member actually served a given call.
func (f *fallback) Model() string {
	if len(f.models) == 0 {
		return ""
	}
	return f.models[0].Model()
}

// FromCatwalk builds an evaluation model from a catwalk provider entry using
// its default evaluation model (or modelID when non-empty). API keys of the
// form $ENV are resolved from the environment.
func FromCatwalk(ctx context.Context, provider catwalk.Provider, modelID string) (fantasy.EvaluationModel, error) {
	if modelID == "" {
		modelID = provider.DefaultEvaluationModelID
	}
	apiKey := resolveEnv(provider.APIKey)
	var p fantasy.Provider
	var err error
	switch provider.Type {
	case catwalk.TypeVercel:
		opts := []vercel.Option{vercel.WithAPIKey(apiKey)}
		if provider.APIEndpoint != "" {
			opts = append(opts, vercel.WithBaseURL(provider.APIEndpoint))
		}
		if len(provider.DefaultHeaders) > 0 {
			opts = append(opts, vercel.WithHeaders(provider.DefaultHeaders))
		}
		p, err = vercel.New(opts...)
	case catwalk.TypeTypeSafe:
		opts := []typesafe.Option{typesafe.WithAPIKey(apiKey)}
		if provider.APIEndpoint != "" {
			opts = append(opts, typesafe.WithBaseURL(provider.APIEndpoint))
		}
		if len(provider.DefaultHeaders) > 0 {
			opts = append(opts, typesafe.WithHeaders(provider.DefaultHeaders))
		}
		p, err = typesafe.New(opts...)
	case catwalk.TypeKev:
		opts := []kev.Option{kev.WithAutoLibraries()}
		if modelID != "" {
			opts = append(opts, kev.WithCheckpoint(kev.Checkpoint(modelID)))
		}
		p, err = kev.New(opts...)
	default:
		return nil, fmt.Errorf("%w: %q (%s)", ErrUnsupportedProvider, provider.ID, provider.Type)
	}
	if err != nil {
		return nil, err
	}
	ep, ok := p.(fantasy.EvaluationProvider)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedProvider, provider.ID)
	}
	return ep.EvaluationModel(ctx, modelID)
}

func resolveEnv(value string) string {
	if name, ok := strings.CutPrefix(value, "$"); ok {
		return os.Getenv(name)
	}
	return value
}
