package gojev

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/taigrr/catwalk/pkg/catwalk"
	"github.com/taigrr/fantasy"
)

type stubModel struct {
	name string
	resp *fantasy.EvaluationResponse
	err  error
}

func (s stubModel) Evaluate(context.Context, fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	return s.resp, s.err
}
func (s stubModel) Provider() string { return s.name }
func (s stubModel) Model() string    { return s.name + "-model" }

func TestFallback(t *testing.T) {
	ok := &fantasy.EvaluationResponse{Model: "ok"}
	fb := Fallback(
		stubModel{name: "a", err: errors.New("down")},
		stubModel{name: "b", resp: ok},
	)
	resp, err := fb.Evaluate(context.Background(), fantasy.EvaluationCall{})
	if err != nil || resp != ok {
		t.Fatalf("resp=%v err=%v", resp, err)
	}
	if fb.Provider() != "a,b" || fb.Model() != "a-model" {
		t.Fatalf("provider=%q model=%q", fb.Provider(), fb.Model())
	}

	_, err = Fallback(stubModel{name: "a", err: errors.New("x")}, stubModel{name: "b", err: errors.New("y")}).Evaluate(context.Background(), fantasy.EvaluationCall{})
	if err == nil || err.Error() != "a/a-model: x\nb/b-model: y" {
		t.Fatalf("err = %v", err)
	}
	if _, err := Fallback().Evaluate(context.Background(), fantasy.EvaluationCall{}); !errors.Is(err, ErrNoEvaluators) {
		t.Fatalf("err = %v", err)
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
}

func TestAskBoolRateAndErrors(t *testing.T) {
	model := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeBool, Probability: 0.8},
	}}}
	p, err := AskBool(context.Background(), model, "state", "Is it?")
	if err != nil || p != 0.8 || !p.Yes() || !p.Above(0.75) || p.Above(0.9) {
		t.Fatalf("p=%v err=%v", p, err)
	}
	if _, err := Ask(context.Background(), stubModel{resp: &fantasy.EvaluationResponse{}}, "s", Bool("?")); err == nil {
		t.Fatal("expected missing answer error")
	}

	scoreModel := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeScore, Score: 1.6, LevelProbabilities: []float64{0.1, 0.2, 0.7}},
	}}}
	rating, err := Rate(context.Background(), scoreModel, "s", "how bad?", "low", "medium", "high")
	if err != nil {
		t.Fatal(err)
	}
	if rating.Level() != "high" || !rating.AtLeast("medium") || rating.AtLeast("high") || rating.AtLeast("nope") {
		t.Fatalf("rating = %+v", rating)
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
	if _, err := FromCatwalk(context.Background(), catwalk.Provider{Type: catwalk.TypeOpenAI}, ""); !errors.Is(err, ErrUnsupportedProvider) {
		t.Fatalf("err = %v", err)
	}
}
