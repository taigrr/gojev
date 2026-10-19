package gojev

import (
	"context"
	"testing"

	"github.com/taigrr/fantasy"
)

func TestRateOwnsSlices(t *testing.T) {
	levels := []string{"low", "high"}
	probabilities := []float64{0.25, 0.75}
	model := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeScore, Score: 0.75, LevelProbabilities: probabilities},
	}}}
	rating, err := Rate(context.Background(), model, "state", "Urgency?", levels...)
	if err != nil {
		t.Fatal(err)
	}

	// Reusing caller/model buffers must not change an existing decision.
	levels[1] = "changed"
	probabilities[1] = 0
	if rating.Level() != "high" || !rating.AtLeast("low") || rating.Probabilities[1] != 0.75 {
		t.Errorf("rating changed through source slices: %+v", rating)
	}

	// Editing a returned rating must not corrupt caller or model buffers.
	rating.Levels[0] = "edited"
	rating.Probabilities[0] = 1
	if levels[0] != "low" || probabilities[0] != 0.25 {
		t.Errorf("rating mutation reached source buffers: levels=%v probabilities=%v", levels, probabilities)
	}
}

func TestRatePreservesAbsentProbabilities(t *testing.T) {
	model := stubModel{resp: &fantasy.EvaluationResponse{Answers: map[string]fantasy.EvaluationAnswer{
		"q": {Type: fantasy.EvaluationQuestionTypeScore, Score: 1},
	}}}
	rating, err := Rate(context.Background(), model, "state", "Urgency?", "low", "high")
	if err != nil {
		t.Fatal(err)
	}
	if rating.Probabilities != nil || rating.Level() != "high" {
		t.Fatalf("rating = %+v", rating)
	}
}
