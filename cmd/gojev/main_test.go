package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/taigrr/fantasy"
)

func TestParseQuestion(t *testing.T) {
	name, q, err := parseQuestion("route=choice:Route this;billing=payments,shipping=")
	if err != nil {
		t.Fatal(err)
	}
	if name != "route" || q.Type != fantasy.EvaluationQuestionTypeChoice || q.Options["billing"] != "payments" {
		t.Fatalf("got %s %+v", name, q)
	}
	if _, has := q.Options["shipping"]; !has {
		t.Fatal("expected shipping option")
	}

	_, q, err = parseQuestion("u=score:How urgent?;low,medium,high")
	if err != nil || len(q.Levels) != 3 {
		t.Fatalf("score parse: %+v %v", q, err)
	}

	_, q, err = parseQuestion("r=bool:Refund?;asks for money,does not")
	if err != nil || q.TrueCriteria != "asks for money" || q.FalseCriteria != "does not" {
		t.Fatalf("bool parse: %+v %v", q, err)
	}

	// Whitespace around list items is trimmed.
	_, q, err = parseQuestion("u=score: How urgent? ; low, medium , high")
	if err != nil || len(q.Levels) != 3 || q.Levels[1] != "medium" || q.Instructions != "How urgent?" {
		t.Fatalf("trimmed score parse: %+v %v", q, err)
	}
	_, q, err = parseQuestion("r=choice:Route; billing = payments , shipping")
	if err != nil || q.Options["billing"] != "payments" {
		t.Fatalf("trimmed choice parse: %+v %v", q, err)
	}
	if _, has := q.Options["shipping"]; !has {
		t.Fatal("expected shipping option")
	}

	for _, bad := range []string{
		"noname", "x=nope:hi", "x=bool", "x=bool:", "=bool:hi",
		"c=choice:pick", "c=choice:pick;=desc", "c=choice:pick;a,a",
		"s=score:rate;only", "s=score:rate;low,low",
	} {
		if _, _, err := parseQuestion(bad); err == nil {
			t.Errorf("expected error for %q", bad)
		}
	}

	if _, err := parseQuestions([]string{"a=bool:x", "a=bool:y"}); err == nil {
		t.Error("expected duplicate name error")
	}
	if _, err := buildState(stateStdin, false, nil); err == nil {
		t.Error("expected nil stdin error")
	}
}

func TestRunAgainstKevDialect(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"model":"kev-latest","answers":{"refund":{"type":"noul","noul":0.9}},"usage":{"input_tokens":5,"output_tokens":1}}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	err := run([]string{
		"-provider", "kev-server", "-base-url", server.URL,
		"-state", "-", "-q", "refund=bool:Refund?",
	}, strings.NewReader("I want my money back"), &out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "refund: true (p=0.900)") {
		t.Fatalf("output = %s", out.String())
	}
}

func TestRunJSONState(t *testing.T) {
	var gotState string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(r.Body)
		gotState = buf.String()
		_, _ = w.Write([]byte(`{"model":"m","answers":{},"usage":{}}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	err := run([]string{
		"-provider", "typesafe", "-base-url", server.URL, "-json",
		"-state", `{"a":1}`, "-json-state", "-q", "x=bool:?",
	}, nil, &out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(gotState, `"state":{"a":1}`) {
		t.Fatalf("state = %s", gotState)
	}
}

func TestRunCloudflare(t *testing.T) {
	var gotPath, gotAuth, gotGateway, gotModel string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotAuth, gotGateway = r.URL.Path, r.Header.Get("Authorization"), r.Header.Get("cf-aig-gateway-id")
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		_, _ = w.Write([]byte(`{"result":{"model":"clef-flash","answers":{"urgent":{"type":"noul","noul":0.8}},"usage":{"input_tokens":5,"output_tokens":0}},"success":true,"errors":[],"messages":[]}`))
	}))
	defer server.Close()
	t.Setenv(envCloudflareAPIToken, "tok")
	t.Setenv(envCloudflareAccountID, "")
	t.Setenv(envCloudflareWorkersURL, "")
	t.Setenv(envCloudflareGatewayID, "")

	var out bytes.Buffer
	err := run([]string{
		"-provider", "cloudflare", "-base-url", server.URL, "-model", "@cf/cloudflare/clef-flash", "-gateway", "gw",
		"-state", "Checkout is down", "-q", "urgent=bool:Urgent?",
	}, nil, &out)
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/@cf/cloudflare/clef-flash" || gotAuth != "Bearer tok" || gotGateway != "gw" || gotModel != "clef-flash" {
		t.Fatalf("path=%q auth=%q gateway=%q model=%q", gotPath, gotAuth, gotGateway, gotModel)
	}
	if !strings.Contains(out.String(), "urgent: true (p=0.800)") {
		t.Fatalf("output = %s", out.String())
	}

	// Without -base-url an account is required.
	if err := run([]string{"-provider", "cloudflare", "-state", "x", "-q", "a=bool:?"}, nil, nil); err == nil || !strings.Contains(err.Error(), "-account") {
		t.Fatalf("err = %v", err)
	}
	// The run root may also come from the environment.
	t.Setenv(envCloudflareWorkersURL, server.URL)
	if err := run([]string{"-provider", "cloudflare", "-state", "x", "-q", "a=bool:?"}, nil, &out); err != nil {
		t.Fatal(err)
	}
	if gotPath != "/@cf/cloudflare/clef" {
		t.Fatalf("path = %q", gotPath)
	}
}

func TestRunVersion(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"-version"}, nil, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.String(), "gojev ") {
		t.Fatalf("output = %q", out.String())
	}
}

func TestRunErrors(t *testing.T) {
	if err := run([]string{}, nil, nil); err == nil {
		t.Fatal("expected missing state error")
	}
	if err := run([]string{"-state", "x"}, nil, nil); err == nil {
		t.Fatal("expected missing question error")
	}
	if err := run([]string{"-state", "x", "-q", "a=bool:?", "-provider", "nope"}, nil, nil); err == nil {
		t.Fatal("expected unknown provider error")
	}
}

func TestPrintResponse(t *testing.T) {
	confidence := 0.75
	resp := &fantasy.EvaluationResponse{
		Model: "jev-test",
		Answers: map[string]fantasy.EvaluationAnswer{
			"route": {
				Type:          fantasy.EvaluationQuestionTypeChoice,
				Choice:        "billing",
				Confidence:    &confidence,
				Probabilities: map[string]float64{"shipping": 0.2, "billing": 0.7, "returns": 0.7},
			},
			"refund": {
				Type:        fantasy.EvaluationQuestionTypeBool,
				Probability: 0.9,
			},
			"urgency": {
				Type:               fantasy.EvaluationQuestionTypeScore,
				Score:              1.6,
				Levels:             []string{"low", "medium", "high"},
				LevelProbabilities: []float64{0.1, 0.3, 0.6},
				Confidence:         &confidence,
			},
			"raw": {
				Type: "mystery",
			},
		},
		Usage: fantasy.EvaluationUsage{InputTokens: 9, OutputTokens: 2},
	}

	var out bytes.Buffer
	printResponse(&out, resp)
	got := out.String()
	want := strings.Join([]string{
		"model: jev-test",
		"raw: (unrecognised answer type \"mystery\")",
		"refund: true (p=0.900)",
		"route: billing (confidence=0.750)",
		"  billing: 0.700",
		"  returns: 0.700",
		"  shipping: 0.200",
		"urgency: 1.60 high (confidence=0.750)",
		"  0 low: 0.100",
		"  1 medium: 0.300",
		"  2 high: 0.600",
		"usage: in=9 out=2",
		"",
	}, "\n")
	if got != want {
		t.Fatalf("output =\n%s\nwant =\n%s", got, want)
	}
}
