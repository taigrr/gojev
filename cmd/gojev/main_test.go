package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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

func TestRunInputRequest(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"model":"m","answers":{"refund":{"type":"noul","noul":0.9},"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"shipping":0.1}},"urgency":{"type":"score","score":1.5,"legend":{"0":"low","1":"mid","2":"high"},"probabilities":{"0":0,"1":0.5,"2":0.5}}},"usage":{"input_tokens":1,"output_tokens":0}}`))
	}))
	defer server.Close()

	// System One wire shape (noul/criteria) and fantasy's own shape
	// (bool/options/levels) are both accepted.
	input := `{
		"state": {"ticket": "charged twice"},
		"questions": {
			"refund": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "asks for money back"}},
			"team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "payments", "shipping": null}},
			"urgency": {"type": "score", "instructions": "Urgency?", "levels": ["low", "mid", "high"]}
		}
	}`
	var out bytes.Buffer
	err := run([]string{"-provider", "typesafe", "-base-url", server.URL, "-input", "-", "-json"}, strings.NewReader(input), &out)
	if err != nil {
		t.Fatal(err)
	}
	if state := gotBody["state"].(map[string]any); state["ticket"] != "charged twice" {
		t.Fatalf("state = %v", gotBody["state"])
	}
	questions := gotBody["questions"].(map[string]any)
	if criteria := questions["refund"].(map[string]any)["criteria"].(map[string]any); criteria["true"] != "asks for money back" {
		t.Fatalf("refund criteria = %v", criteria)
	}
	if criteria := questions["team"].(map[string]any)["criteria"].(map[string]any); criteria["billing"] != "payments" || criteria["shipping"] != nil {
		t.Fatalf("team criteria = %v", criteria)
	}
	if levels := questions["urgency"].(map[string]any)["criteria"].([]any); len(levels) != 3 || levels[2] != "high" {
		t.Fatalf("urgency criteria = %v", levels)
	}
	var resp struct {
		Answers map[string]struct {
			Type               string    `json:"type"`
			Probability        float64   `json:"probability"`
			Choice             string    `json:"choice"`
			Score              float64   `json:"score"`
			LevelProbabilities []float64 `json:"level_probabilities"`
		} `json:"answers"`
	}
	if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
		t.Fatalf("output not JSON: %v\n%s", err, out.String())
	}
	if resp.Answers["refund"].Probability != 0.9 || resp.Answers["team"].Choice != "billing" || resp.Answers["urgency"].Score != 1.5 || len(resp.Answers["urgency"].LevelProbabilities) != 3 {
		t.Fatalf("answers = %+v", resp.Answers)
	}

	// A file works too, and -input is exclusive with -state/-q.
	file := filepath.Join(t.TempDir(), "req.json")
	if err := os.WriteFile(file, []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-provider", "typesafe", "-base-url", server.URL, "-input", file}, nil, &out); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-input", "-", "-state", "x"}, strings.NewReader(input), nil); err == nil {
		t.Fatal("expected exclusivity error")
	}
	for _, bad := range []string{
		`{"questions":{"q":{"type":"noul"}}}`,
		`{"state":"x"}`,
		`{"state":"x","questions":{"q":{"type":"mystery"}}}`,
		`{"state":"x","questions":{"q":{"type":"choice"}}}`,
		`{"state":"x","questions":{"q":{"type":"score","criteria":["only"]}}}`,
		`{"state":"x","questions":{"q":{"type":"noul","criteria":["not","an","object"]}}}`,
		`not json`,
	} {
		if err := run([]string{"-provider", "typesafe", "-base-url", server.URL, "-input", "-"}, strings.NewReader(bad), nil); err == nil {
			t.Fatalf("expected error for %s", bad)
		}
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
