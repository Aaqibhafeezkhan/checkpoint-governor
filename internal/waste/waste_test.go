package waste

import (
	"math"
	"strings"
	"testing"
)

// The formula is the claim: waste fraction = T_write / (interval × t_step). If this drifts, every
// number the table prints is wrong, so it is pinned directly.
func TestTheWasteFractionIsWriteTimeOverInterval(t *testing.T) {
	a := Assumptions{Hours: 4, BytesPerParam: 16, ThroughputGBs: 2, StepsPerEpoch: 1000}
	s := Scale{Name: "1B / 8 GPU", Params: 1e9, GPUs: 8, StepMS: 1000}
	fw := Frameworks[0]

	r, err := Run(a, fw, s)
	if err != nil {
		t.Fatal(err)
	}
	if r.IntervalSteps != 500 {
		t.Fatalf("interval = %d, want the framework default 500", r.IntervalSteps)
	}
	if want := 8.0; math.Abs(r.WriteSeconds-want) > 1e-9 {
		t.Fatalf("write seconds = %v, want %v (16 GB at 2 GB/s)", r.WriteSeconds, want)
	}
	if want := 8.0 / 500.0; math.Abs(r.WasteFraction-want) > 1e-9 {
		t.Fatalf("waste fraction = %v, want %v", r.WasteFraction, want)
	}
	if want := 0.016 * 8 * 4; math.Abs(r.WasteGPUHours-want) > 1e-9 {
		t.Fatalf("waste gpu-hours = %v, want %v", r.WasteGPUHours, want)
	}
	if r.Writes != 28 {
		t.Fatalf("writes = %d, want 28", r.Writes)
	}
	if want := 28 * 16e9; math.Abs(r.StorageBytes-want) > 1 {
		t.Fatalf("storage = %v, want %v", r.StorageBytes, want)
	}
}

func TestAWriteLongerThanTheIntervalIsInfeasible(t *testing.T) {
	a := Assumptions{Hours: 4, BytesPerParam: 16, ThroughputGBs: 2, StepsPerEpoch: 1000}
	s := Scale{Name: "huge", Params: 1e12, GPUs: 512, StepMS: 1000}
	r, err := Run(a, Frameworks[0], s)
	if err != nil {
		t.Fatal(err)
	}
	if r.Feasible {
		t.Fatalf("a %vs write against a 500s interval must be infeasible", r.WriteSeconds)
	}
	if r.Note == "" {
		t.Fatal("an infeasible row must say why")
	}
}

func TestPerEpochDefaultsUseTheGivenEpoch(t *testing.T) {
	a := Assumptions{Hours: 4, BytesPerParam: 16, ThroughputGBs: 2, StepsPerEpoch: 1000}
	s := Scale{Name: "1B / 8 GPU", Params: 1e9, GPUs: 8, StepMS: 1000}
	for _, name := range []string{"lightning", "composer"} {
		r, err := Run(a, find(t, name), s)
		if err != nil {
			t.Fatal(err)
		}
		if r.IntervalSteps != 1000 {
			t.Fatalf("%s interval = %d, want one epoch (1000)", name, r.IntervalSteps)
		}
	}
	l, _ := Run(a, find(t, "lightning"), s)
	c, _ := Run(a, find(t, "composer"), s)
	if l.StorageBytes != 16e9 {
		t.Fatalf("lightning storage = %v, want one 16 GB copy", l.StorageBytes)
	}
	if c.StorageBytes <= l.StorageBytes {
		t.Fatalf("composer keeps all copies: %v should exceed %v", c.StorageBytes, l.StorageBytes)
	}
}

func TestMegatronHasNoDefault(t *testing.T) {
	a := Assumptions{Hours: 4, BytesPerParam: 16, ThroughputGBs: 2, StepsPerEpoch: 1000}
	r, err := Run(a, find(t, "megatron"), Scale{Name: "7B / 64 GPU", Params: 7e9, GPUs: 64, StepMS: 2000})
	if err != nil {
		t.Fatal(err)
	}
	if r.IntervalSteps != 0 {
		t.Fatalf("megatron interval = %d, want 0 (no default)", r.IntervalSteps)
	}
	if r.Note == "" || r.Source == "" {
		t.Fatal("a row without a note and a source is a rumour")
	}
}

func TestMegatronExplicitIntervalPricesEveryScale(t *testing.T) {
	fw, err := WithExplicitInterval(find(t, "megatron"), 1000)
	if err != nil {
		t.Fatal(err)
	}
	if fw.IntervalSteps(0) != 1000 {
		t.Fatalf("interval = %d, want 1000", fw.IntervalSteps(0))
	}
	if fw.Source != "explicit --save-interval=1000" {
		t.Fatalf("source = %q", fw.Source)
	}
	if fw.Keeps != "" {
		t.Fatalf("explicit interval must not invent retention, got %q", fw.Keeps)
	}
	a := Assumptions{Hours: 4, BytesPerParam: 16, ThroughputGBs: 2, StepsPerEpoch: 1000}
	for _, scale := range Scales {
		r, err := Run(a, fw, scale)
		if err != nil {
			t.Fatal(err)
		}
		if r.IntervalSteps != 1000 || r.WasteFraction <= 0 || !strings.Contains(r.Source, "--save-interval=1000") {
			t.Fatalf("unexpected explicit Megatron row for %s: %+v", scale.Name, r)
		}
		if r.Keeps != "" {
			t.Fatalf("retention should remain unspecified for %s", scale.Name)
		}
	}
}

func TestExplicitIntervalMustBePositive(t *testing.T) {
	if _, err := WithExplicitInterval(find(t, "megatron"), 0); err == nil {
		t.Fatal("expected non-positive interval to be rejected")
	}
}

func TestEveryFrameworkNamesItsSource(t *testing.T) {
	for _, f := range Frameworks {
		if f.Source == "" {
			t.Errorf("%s has no source", f.Name)
		}
	}
}

func find(t *testing.T, name string) Framework {
	t.Helper()
	for _, f := range Frameworks {
		if f.Name == name {
			return f
		}
	}
	t.Fatalf("no framework %q", name)
	return Framework{}
}
