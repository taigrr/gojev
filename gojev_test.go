package gojev

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/taigrr/catwalk/pkg/catwalk"
	"github.com/taigrr/fantasy"
)

type stubModel struct {
	name    string
	resp    *fantasy.EvaluationResponse
	err     error
	loadErr error
	calls   *int
	loaded  *bool
}

func (s stubModel) Evaluate(context.Context, fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if s.calls != nil {
		*s.calls++
	}
	return s.resp, s.err
}
func (s stubModel) Provider() string { return s.name }
func (s stubModel) Model() string    { return s.name + "-model" }
func (s stubModel) Load(context.Context) error {
	if s.loadErr != nil {
		return s.loadErr
	}
	if s.loaded != nil {
		*s.loaded = true
	}
	return nil
}

var validCall = fantasy.EvaluationCall{State: "s", Questions: Questions{"q": Bool("?")}}

func TestFallback(t *testing.T) {
	ok := &fantasy.EvaluationResponse{Model: "ok"}
	fb := Fallback(
		nil,
		stubModel{name: "a", err: errors.New("down")},
		stubModel{name: "b", resp: ok},
	)
	resp, err := fb.Evaluate(context.Background(), validCall)
	if err != nil || resp != ok {
		t.Fatalf("resp=%v err=%v", resp, err)
	}
	if fb.Provider() != "a,b" || fb.Model() != "a-model" {
		t.Fatalf("provider=%q model=%q", fb.Provider(), fb.Model())
	}

	_, err = Fallback(stubModel{name: "a", err: errors.New("x")}, stubModel{name: "b", err: errors.New("y")}).Evaluate(context.Background(), validCall)
	if err == nil || err.Error() != "a/a-model: x\nb/b-model: y" {
		t.Fatalf("err = %v", err)
	}
	if _, err := Fallback().Evaluate(context.Background(), validCall); !errors.Is(err, ErrNoEvaluators) {
		t.Fatalf("err = %v", err)
	}

	// Invalid calls fail fast without touching any member.
	calls := 0
	if _, err := Fallback(stubModel{name: "a", calls: &calls, resp: ok}).Evaluate(context.Background(), fantasy.EvaluationCall{}); err == nil || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}

	// A (nil, nil) member is an error, not a success.
	if _, err := Fallback(stubModel{name: "a"}).Evaluate(context.Background(), validCall); !errors.Is(err, ErrNilResponse) {
		t.Fatalf("err = %v", err)
	}

	// A cancelled context stops the chain and keeps attribution.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	calls = 0
	_, err = Fallback(stubModel{name: "a", err: ctx.Err()}, stubModel{name: "b", calls: &calls, resp: ok}).Evaluate(ctx, validCall)
	if !errors.Is(err, context.Canceled) || calls != 0 || !strings.HasPrefix(err.Error(), "a/a-model:") {
		t.Fatalf("err=%v calls=%d", err, calls)
	}

	// Load is forwarded to members and only fails when no member loaded.
	loaded := false
	if err := fb.(interface{ Load(context.Context) error }).Load(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := Fallback(stubModel{name: "k", loaded: &loaded}).(interface{ Load(context.Context) error }).Load(context.Background()); err != nil || !loaded {
		t.Fatalf("loaded=%v err=%v", loaded, err)
	}
	loaded = false
	partial := Fallback(stubModel{name: "k", loadErr: errors.New("offline")}, stubModel{name: "c", loaded: &loaded})
	if err := partial.(interface{ Load(context.Context) error }).Load(context.Background()); err != nil || !loaded {
		t.Fatalf("partial load: loaded=%v err=%v", loaded, err)
	}
	allDown := Fallback(stubModel{name: "k", loadErr: errors.New("offline")})
	if err := allDown.(interface{ Load(context.Context) error }).Load(context.Background()); err == nil || !strings.Contains(err.Error(), "k/k-model: offline") {
		t.Fatalf("all-down load err = %v", err)
	}
}

type verdict string

const (
	allow verdict = "allow"
	ask   verdict = "ask"
	deny  verdict = "deny"
)

func TestClassifyAndDecision(t *testing.T) {
	model := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeChoice, Choice: "allow", Probabilities: map[string]float64{"allow": 0.7, "ask": 0.25, "deny": 0.05}},
	}}}
	decision, err := Classify(context.Background(), model, "rm foo.txt", "Permit?", []verdict{allow, ask, deny}, map[verdict]string{allow: "safe"})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Selected != allow || decision.Probabilities[ask] != 0.25 {
		t.Fatalf("decision = %+v", decision)
	}
	if m := decision.Margin(); m < 0.449 || m > 0.451 {
		t.Fatalf("margin = %v", m)
	}
	if v, ok := decision.Above(0.6, 0.4); !ok || v != allow {
		t.Fatalf("Above = %v %v", v, ok)
	}
	if _, ok := decision.Above(0.8, 0); ok {
		t.Fatal("expected threshold rejection")
	}
	if _, ok := decision.Above(0.6, 0.5); ok {
		t.Fatal("expected margin rejection")
	}
	single := Decision[verdict]{Selected: allow, Probabilities: map[verdict]float64{allow: 1}}
	if single.Margin() != 1 {
		t.Fatalf("single margin = %v", single.Margin())
	}

	// A zero Decision never passes, even with zero thresholds.
	if _, ok := (Decision[verdict]{}).Above(0, 0); ok {
		t.Fatal("zero decision passed Above")
	}
	nan := Decision[verdict]{Selected: allow, Probabilities: map[verdict]float64{allow: math.NaN()}}
	if _, ok := nan.Above(0, 0); ok {
		t.Fatal("NaN passed Above")
	}

	// Selections and probabilities outside the offered set are errors.
	unknown := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeChoice, Choice: "Allow", Probabilities: map[string]float64{"Allow": 1}},
	}}}
	if _, err := Classify(context.Background(), unknown, "s", "?", []verdict{allow, deny}, nil); !errors.Is(err, ErrUnknownChoice) {
		t.Fatalf("err = %v", err)
	}
	extra := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeChoice, Choice: "allow", Probabilities: map[string]float64{"allow": 0.6, "maybe": 0.4}},
	}}}
	if _, err := Classify(context.Background(), extra, "s", "?", []verdict{allow, deny}, nil); !errors.Is(err, ErrUnknownChoice) {
		t.Fatalf("err = %v", err)
	}

	// Offered options absent from the response are present with 0.
	sparse := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeChoice, Choice: "allow", Probabilities: map[string]float64{"allow": 1}},
	}}}
	decision, err = Classify(context.Background(), sparse, "s", "?", []verdict{allow, deny}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if p, ok := decision.Probabilities[deny]; !ok || p != 0 {
		t.Fatalf("deny probability = %v %v", p, ok)
	}

	// A differently typed answer is rejected, as is a choice with no
	// probabilities.
	wrongType := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeBool, Probability: 1},
	}}}
	if _, err := Classify(context.Background(), wrongType, "s", "?", []verdict{allow, deny}, nil); !errors.Is(err, ErrAnswerTypeMismatch) {
		t.Fatalf("err = %v", err)
	}
	noProbs := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeChoice, Choice: "allow"},
	}}}
	if _, err := Classify(context.Background(), noProbs, "s", "?", []verdict{allow, deny}, nil); !errors.Is(err, ErrAnswerTypeMismatch) {
		t.Fatalf("err = %v", err)
	}
}

func TestAskBoolRateAndErrors(t *testing.T) {
	model := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeBool, Probability: 0.8},
	}}}
	p, err := AskBool(context.Background(), model, "state", "Is it?")
	if err != nil || p != 0.8 || !p.Yes() || !p.Above(0.75) || p.Above(0.9) {
		t.Fatalf("p=%v err=%v", p, err)
	}
	if _, err := Ask(context.Background(), stubModel{resp: &fantasy.EvaluationResponse{}}, "s", Bool("?")); !errors.Is(err, ErrMissingAnswer) {
		t.Fatalf("err = %v", err)
	}
	if _, err := Ask(context.Background(), stubModel{}, "s", Bool("?")); !errors.Is(err, ErrNilResponse) {
		t.Fatalf("err = %v", err)
	}

	scoreModel := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeScore, Score: 1.6, Levels: []string{"LOW", "MEDIUM", "HIGH"}, LevelProbabilities: []float64{0.1, 0.2, 0.7}},
	}}}
	rating, err := Rate(context.Background(), scoreModel, "s", "how bad?", "low", "medium", "high")
	if err != nil {
		t.Fatal(err)
	}
	// Caller's labels win over the model's echo.
	if rating.Level() != "high" || rating.LevelIndex() != 2 || !rating.AtLeast("medium") || rating.AtLeast("high") || rating.AtLeast("nope") {
		t.Fatalf("rating = %+v", rating)
	}
	for score, want := range map[float64]int{-3: 0, 0.49: 0, 0.5: 1, 1.5: 2, 9: 2, math.Inf(1): 2, math.Inf(-1): 0, math.NaN(): -1} {
		if got := (Rating{Score: score, Levels: rating.Levels}).LevelIndex(); got != want {
			t.Errorf("LevelIndex(%v) = %d, want %d", score, got, want)
		}
	}
	if (Rating{}).LevelIndex() != -1 || (Rating{}).Level() != "" {
		t.Fatal("empty rating should have no level")
	}
	badShape := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeScore, Score: 1, LevelProbabilities: []float64{1}},
	}}}
	if _, err := Rate(context.Background(), badShape, "s", "?", "a", "b"); !errors.Is(err, ErrAnswerTypeMismatch) {
		t.Fatalf("err = %v", err)
	}
	badEcho := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeScore, Score: 1, Levels: []string{"x", "y", "z"}},
	}}}
	if _, err := Rate(context.Background(), badEcho, "s", "?", "a", "b"); !errors.Is(err, ErrAnswerTypeMismatch) {
		t.Fatalf("err = %v", err)
	}
}

func TestFromCatwalk(t *testing.T) {
	var gotPath, gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotAuth = r.URL.Path, r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"model":"m","answers":{"q":{"type":"noul","noul":0.6}},"usage":{}}`))
	}))
	defer server.Close()
	t.Setenv("TEST_TS_KEY", "sekrit")

	model, err := FromCatwalk(context.Background(), catwalk.Provider{
		ID:                       catwalk.InferenceProviderTypeSafe,
		Type:                     catwalk.TypeTypeSafe,
		APIKey:                   "$TEST_TS_KEY",
		APIEndpoint:              server.URL,
		DefaultEvaluationModelID: "jev-latest",
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	if model.Model() != "jev-latest" {
		t.Fatalf("model = %q", model.Model())
	}
	answer, err := Ask(context.Background(), model, "x", Bool("?"))
	if err != nil || !answer.Yes() {
		t.Fatalf("answer=%+v err=%v", answer, err)
	}
	if gotPath != "/v1/systemone" || gotAuth != "Bearer sekrit" {
		t.Fatalf("path=%q auth=%q", gotPath, gotAuth)
	}

	vercelModel, err := FromCatwalk(context.Background(), catwalk.Provider{
		Type: catwalk.TypeVercel, APIKey: "k", DefaultEvaluationModelID: "typesafe-ai/jev",
	}, "")
	if err != nil || vercelModel.Model() != "typesafe-ai/jev" || vercelModel.Provider() != "vercel" {
		t.Fatalf("vercel model=%v err=%v", vercelModel, err)
	}

	kevModel, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeKev, DefaultEvaluationModelID: "kev-0.8b"}, "")
	if err != nil || kevModel.Model() != "kev-0.8b" || kevModel.Provider() != "kev" {
		t.Fatalf("kev model=%v err=%v", kevModel, err)
	}
	if _, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeKev}, "kev-latest"); err != nil {
		t.Fatalf("kev-latest err = %v", err)
	}
	if _, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeOpenAI}, ""); !errors.Is(err, ErrUnsupportedProvider) {
		t.Fatalf("err = %v", err)
	}
	t.Setenv("GOJEV_UNSET_KEY", "")
	if _, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeTypeSafe, APIKey: "$GOJEV_UNSET_KEY"}, ""); !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
	if _, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeVercel}, ""); !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("err = %v", err)
	}
	// A self-hosted TypeSafe-compatible endpoint (local Kev) needs no key.
	localKev, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeTypeSafe, APIEndpoint: server.URL, DefaultEvaluationModelID: "kev-latest"}, "")
	if err != nil || localKev.Model() != "kev-latest" {
		t.Fatalf("local kev model=%v err=%v", localKev, err)
	}
}
