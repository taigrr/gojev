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
	"io"
	"math"
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

// ErrNilResponse is returned when a model yields neither a response nor an
// error.
var ErrNilResponse = errors.New("gojev: model returned nil response")

// ErrMissingAnswer is returned when the response lacks the requested answer.
var ErrMissingAnswer = errors.New("gojev: response missing answer")

// ErrAnswerTypeMismatch is returned when the answer's type differs from the
// question that was asked.
var ErrAnswerTypeMismatch = errors.New("gojev: answer type does not match question")

// ErrUnknownChoice is returned by Classify when the model selects, or
// assigns probability to, an option the caller did not offer.
var ErrUnknownChoice = errors.New("gojev: model returned an option that was not offered")

// ErrMissingAPIKey is returned by FromCatwalk when a hosted provider's API
// key resolves to the empty string.
var ErrMissingAPIKey = errors.New("gojev: provider API key is empty")

// Evaluate is a convenience wrapper around model.Evaluate. A model that
// returns neither a response nor an error yields ErrNilResponse.
func Evaluate(ctx context.Context, model fantasy.EvaluationModel, state any, questions Questions) (*fantasy.EvaluationResponse, error) {
	resp, err := model.Evaluate(ctx, fantasy.EvaluationCall{State: state, Questions: questions})
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, ErrNilResponse
	}
	return resp, nil
}

// Ask evaluates a single question and returns its answer. The answer's type
// is checked against the question; a mismatch is ErrAnswerTypeMismatch.
func Ask(ctx context.Context, model fantasy.EvaluationModel, state any, question fantasy.EvaluationQuestion) (fantasy.EvaluationAnswer, error) {
	const key = "q"
	resp, err := Evaluate(ctx, model, state, Questions{key: question})
	if err != nil {
		return fantasy.EvaluationAnswer{}, err
	}
	answer, ok := resp.Answers[key]
	if !ok {
		return fantasy.EvaluationAnswer{}, ErrMissingAnswer
	}
	if answer.Type != question.Type {
		return fantasy.EvaluationAnswer{}, fmt.Errorf("%w: asked %q, got %q", ErrAnswerTypeMismatch, question.Type, answer.Type)
	}
	return answer, nil
}

// Decision is a typed Choice result.
type Decision[T ~string] struct {
	Selected      T
	Probabilities map[T]float64
	Confidence    *float64
}

// Margin is the gap between the selected option's probability and the
// highest probability among the other options. It is 0 when Selected is
// absent from Probabilities, and equals P(Selected) when it is the only
// option.
func (d Decision[T]) Margin() float64 {
	selected, ok := d.Probabilities[d.Selected]
	if !ok {
		return 0
	}
	best := 0.0
	for option, p := range d.Probabilities {
		if option != d.Selected && p > best {
			best = p
		}
	}
	return selected - best
}

// Above returns the selected option only if its probability is at least
// threshold and the margin over the runner-up is at least minMargin. A
// selection with no recorded probability (or a NaN probability) never
// passes, regardless of thresholds.
func (d Decision[T]) Above(threshold, minMargin float64) (T, bool) {
	var zero T
	p, ok := d.Probabilities[d.Selected]
	if !ok || !(p >= threshold) || !(d.Margin() >= minMargin) {
		return zero, false
	}
	return d.Selected, true
}

// Classify asks a Choice question over the given typed options and returns a
// typed Decision. Descriptions may be nil or omit options. The model's
// selection and every option it assigns probability to must be one of the
// offered options; otherwise ErrUnknownChoice is returned. Offered options
// missing from the response are recorded with probability 0.
func Classify[T ~string](ctx context.Context, model fantasy.EvaluationModel, state any, instructions string, options []T, descriptions map[T]string) (Decision[T], error) {
	wire := make(map[string]string, len(options))
	offered := make(map[T]struct{}, len(options))
	for _, option := range options {
		wire[string(option)] = descriptions[option]
		offered[option] = struct{}{}
	}
	answer, err := Ask(ctx, model, state, Choice(instructions, wire))
	if err != nil {
		return Decision[T]{}, err
	}
	selected := T(answer.Choice)
	if _, ok := offered[selected]; !ok {
		return Decision[T]{}, fmt.Errorf("%w: %q", ErrUnknownChoice, answer.Choice)
	}
	if len(answer.Probabilities) == 0 {
		return Decision[T]{}, fmt.Errorf("%w: no option probabilities", ErrAnswerTypeMismatch)
	}
	decision := Decision[T]{
		Selected:      selected,
		Probabilities: make(map[T]float64, len(options)),
		Confidence:    answer.Confidence,
	}
	for _, option := range options {
		decision.Probabilities[option] = 0
	}
	for option, p := range answer.Probabilities {
		key := T(option)
		if _, ok := offered[key]; !ok {
			return Decision[T]{}, fmt.Errorf("%w: probability for %q", ErrUnknownChoice, option)
		}
		decision.Probabilities[key] = p
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

// LevelIndex returns the index of the label nearest the expected score
// (rounding half up), clamped to the rubric. It is -1 for an empty rubric
// or a NaN score.
func (r Rating) LevelIndex() int {
	if len(r.Levels) == 0 || math.IsNaN(r.Score) {
		return -1
	}
	last := len(r.Levels) - 1
	switch {
	case r.Score <= 0:
		return 0
	case r.Score >= float64(last):
		return last
	}
	return int(math.Floor(r.Score + 0.5))
}

// Level returns the label nearest the expected score.
func (r Rating) Level() string {
	idx := r.LevelIndex()
	if idx < 0 {
		return ""
	}
	return r.Levels[idx]
}

// AtLeast reports whether the expected score reaches the named level. It is
// a threshold on the raw expected score, so it is stricter than Level():
// a score of 1.6 over three levels has Level() == Levels[2] but is not
// AtLeast(Levels[2]). Unknown labels are never reached.
func (r Rating) AtLeast(level string) bool {
	for i, name := range r.Levels {
		if name == level {
			return r.Score >= float64(i)
		}
	}
	return false
}

// Rate asks a Score question and returns a typed Rating. The caller's
// levels are used as the rubric labels so lookups by name always match
// what was asked, regardless of any normalisation the model applies.
func Rate(ctx context.Context, model fantasy.EvaluationModel, state any, instructions string, levels ...string) (Rating, error) {
	answer, err := Ask(ctx, model, state, Score(instructions, levels...))
	if err != nil {
		return Rating{}, err
	}
	if len(answer.LevelProbabilities) != 0 && len(answer.LevelProbabilities) != len(levels) {
		return Rating{}, fmt.Errorf("%w: %d level probabilities for %d levels", ErrAnswerTypeMismatch, len(answer.LevelProbabilities), len(levels))
	}
	if len(answer.Levels) != 0 && len(answer.Levels) != len(levels) {
		return Rating{}, fmt.Errorf("%w: model echoed %d levels for %d asked", ErrAnswerTypeMismatch, len(answer.Levels), len(levels))
	}
	return Rating{Score: answer.Score, Levels: levels, Probabilities: answer.LevelProbabilities, Confidence: answer.Confidence}, nil
}

// Fallback chains models: each call tries them in order and returns the first
// success. Errors from all members are joined when every member fails.
// Untyped nil members are ignored. All members share the caller's context,
// so a slow first member can exhaust the deadline before later members are
// tried; give the context enough budget for the whole chain. Load and Close
// are forwarded to members that implement them; note that fantasy's
// in-process Kev exposes Close on its provider, not its model.
func Fallback(models ...fantasy.EvaluationModel) fantasy.EvaluationModel {
	kept := make([]fantasy.EvaluationModel, 0, len(models))
	for _, model := range models {
		if model != nil {
			kept = append(kept, model)
		}
	}
	return &fallback{models: kept}
}

type fallback struct {
	models []fantasy.EvaluationModel
}

func (f *fallback) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if len(f.models) == 0 {
		return nil, ErrNoEvaluators
	}
	if err := call.Validate(); err != nil {
		return nil, err
	}
	var errs []error
	for _, model := range f.models {
		resp, err := model.Evaluate(ctx, call)
		if err == nil && resp == nil {
			err = ErrNilResponse
		}
		if err == nil {
			return resp, nil
		}
		errs = append(errs, fmt.Errorf("%s/%s: %w", model.Provider(), model.Model(), err))
		if ctx.Err() != nil {
			break
		}
	}
	return nil, errors.Join(errs...)
}

// Load warms every member that supports loading. Mirroring Evaluate, it
// only fails when every loadable member failed, so an unavailable local
// model does not prevent the chain from serving via another member.
func (f *fallback) Load(ctx context.Context) error {
	var errs []error
	loadable, loaded := 0, 0
	for _, model := range f.models {
		loader, ok := model.(interface{ Load(context.Context) error })
		if !ok {
			continue
		}
		loadable++
		if err := loader.Load(ctx); err != nil {
			errs = append(errs, fmt.Errorf("%s/%s: %w", model.Provider(), model.Model(), err))
			if ctx.Err() != nil {
				break
			}
			continue
		}
		loaded++
	}
	if loadable > 0 && loaded == 0 {
		return errors.Join(errs...)
	}
	return nil
}

// Close releases every member that implements io.Closer, joining errors.
func (f *fallback) Close() error {
	var errs []error
	for _, model := range f.models {
		if closer, ok := model.(io.Closer); ok {
			if err := closer.Close(); err != nil {
				errs = append(errs, fmt.Errorf("%s/%s: %w", model.Provider(), model.Model(), err))
			}
		}
	}
	return errors.Join(errs...)
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
// form $ENV are resolved from the environment; callers whose configuration
// layer supports richer expansion should resolve provider.APIKey before
// calling. Hosted endpoints (vercel, and typesafe without a custom
// APIEndpoint) require a non-empty key and return ErrMissingAPIKey
// otherwise; a typesafe-type entry with its own APIEndpoint, such as a
// local Kev server, may omit the key.
func FromCatwalk(ctx context.Context, provider catwalk.Provider, modelID string) (fantasy.EvaluationModel, error) {
	if modelID == "" {
		modelID = provider.DefaultEvaluationModelID
	}
	apiKey := resolveEnv(provider.APIKey)
	hosted := provider.Type == catwalk.TypeVercel || (provider.Type == catwalk.TypeTypeSafe && provider.APIEndpoint == "")
	if apiKey == "" && hosted {
		return nil, fmt.Errorf("%w: %q", ErrMissingAPIKey, provider.ID)
	}
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
		if modelID != "" && modelID != kev.ModelLatest {
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
