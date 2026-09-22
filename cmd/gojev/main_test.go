package main

import (
	"bytes"
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
