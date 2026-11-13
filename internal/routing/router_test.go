package routing

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func rep(id string, ready bool, queue, inflight int) ReplicaState {
	return ReplicaState{ID: ReplicaID(id), Model: "m", Ready: ready, QueueDepth: queue, InFlight: inflight}
}

func pick(t *testing.T, r Router, req Request, reps []ReplicaState) ReplicaID {
	t.Helper()
	id, err := r.Pick(req, reps)
	if err != nil {
		t.Fatalf("%s.Pick: %v", r.Name(), err)
	}
	return id
}

// ------------------------------------------------------------------- basics

func TestEveryRouterRefusesWhenNoReplicaIsReady(t *testing.T) {
	reps := []ReplicaState{rep("a", false, 0, 0), rep("b", false, 0, 0)}
	for _, name := range RouterNames() {
		r, _ := NewRouter(name)
		if _, err := r.Pick(Request{Model: "m"}, reps); !errors.Is(err, ErrNoReplica) {
			t.Errorf("%s: want ErrNoReplica, got %v", name, err)
		}
	}
}

func TestEveryRouterSkipsUnreadyReplicas(t *testing.T) {
	// A replica that is sleeping, draining or still loading must never be picked:
	// routing to a draining replica is exactly the in-flight request the scheduler
	// drained to avoid.
	reps := []ReplicaState{rep("down", false, 0, 0), rep("up", true, 5, 5)}
	for _, name := range RouterNames() {
		r, _ := NewRouter(name)
		for i := 0; i < 10; i++ {
			if got := pick(t, r, Request{Model: "m"}, reps); got != "up" {
				t.Fatalf("%s picked an unready replica: %s", name, got)
			}
		}
	}
}

func TestRoundRobinSpreadsEvenly(t *testing.T) {
	reps := []ReplicaState{rep("a", true, 0, 0), rep("b", true, 0, 0), rep("c", true, 0, 0)}
	r := NewRoundRobin()
	counts := map[ReplicaID]int{}
	for i := 0; i < 300; i++ {
		counts[pick(t, r, Request{Model: "m"}, reps)]++
	}
	for id, n := range counts {
		if n != 100 {
			t.Fatalf("replica %s got %d of 300, want 100", id, n)
		}
	}
}

func TestRoundRobinIsBlindToLoad(t *testing.T) {
	// The baseline's defining weakness, pinned so the comparison in experiment 5 is
	// against a known quantity.
	reps := []ReplicaState{rep("idle", true, 0, 0), rep("swamped", true, 50, 20)}
	r := NewRoundRobin()
	swamped := 0
	for i := 0; i < 100; i++ {
		if pick(t, r, Request{Model: "m"}, reps) == "swamped" {
			swamped++
		}
	}
	if swamped != 50 {
		t.Fatalf("round-robin sent %d/100 to the swamped replica, want 50", swamped)
	}
}

func TestLeastLoadedPrefersShallowQueues(t *testing.T) {
	reps := []ReplicaState{rep("busy", true, 10, 2), rep("quiet", true, 1, 1)}
	r := NewLeastLoaded()
	for i := 0; i < 20; i++ {
		if got := pick(t, r, Request{Model: "m"}, reps); got != "quiet" {
			t.Fatalf("picked %s, want quiet", got)
		}
	}
}

func TestLeastLoadedWeightsQueueAboveInFlight(t *testing.T) {
	// vLLM's continuous batching absorbs concurrent decodes far better than it
	// absorbs a backlog, so four in-flight requests should be preferred over three
	// queued ones.
	reps := []ReplicaState{
		rep("queued", true, 3, 0),   // load 6
		rep("batching", true, 0, 4), // load 4
	}
	if got := pick(t, NewLeastLoaded(), Request{Model: "m"}, reps); got != "batching" {
		t.Fatalf("picked %s, want batching", got)
	}
}

func TestLeastLoadedBreaksTiesOnRecentTTFT(t *testing.T) {
	reps := []ReplicaState{
		{ID: "slow", Model: "m", Ready: true, QueueDepth: 2, RecentTTFTMillis: 900},
		{ID: "fast", Model: "m", Ready: true, QueueDepth: 2, RecentTTFTMillis: 120},
	}
	if got := pick(t, NewLeastLoaded(), Request{Model: "m"}, reps); got != "fast" {
		t.Fatalf("picked %s, want fast", got)
	}
}

// ---------------------------------------------------------------- prefix hashing

func TestBlockHashesAreCumulativeNotPerBlock(t *testing.T) {
	// Two prompts whose *second* 64-char block is identical but whose first differs
	// share no KV cache at all. Hashing blocks independently would match them.
	shared := strings.Repeat("B", DefaultBlockSize)
	a := BlockHashes(strings.Repeat("A", DefaultBlockSize)+shared, DefaultBlockSize, 16)
	b := BlockHashes(strings.Repeat("Z", DefaultBlockSize)+shared, DefaultBlockSize, 16)

	if len(a) != 2 || len(b) != 2 {
		t.Fatalf("expected 2 blocks each, got %d and %d", len(a), len(b))
	}
	if a[0] == b[0] {
		t.Fatal("different first blocks hashed the same")
	}
	if a[1] == b[1] {
		t.Fatal("cumulative hash matched despite differing first blocks")
	}
}

func TestBlockHashesMatchOnAGenuineSharedPrefix(t *testing.T) {
	sys := strings.Repeat("You are a helpful assistant with tools. ", 8)
	a := BlockHashes(sys+"What is the weather in Paris?", DefaultBlockSize, 32)
	b := BlockHashes(sys+"Book me a flight to Tokyo.", DefaultBlockSize, 32)

	shared := 0
	for i := range a {
		if i >= len(b) || a[i] != b[i] {
			break
		}
		shared++
	}
	if shared < len(sys)/DefaultBlockSize-1 {
		t.Fatalf("shared prefix matched only %d blocks, want about %d", shared, len(sys)/DefaultBlockSize)
	}
}

func TestBlockHashesRespectMaxBlocks(t *testing.T) {
	// A 100KB prompt must not produce 1500 hashes on the request path.
	got := BlockHashes(strings.Repeat("x", 100_000), DefaultBlockSize, 32)
	if len(got) != 32 {
		t.Fatalf("got %d blocks, want 32", len(got))
	}
}

func TestBlockHashesHandlesShortPrompts(t *testing.T) {
	if got := BlockHashes("hi", DefaultBlockSize, 32); len(got) != 0 {
		t.Fatalf("a sub-block prompt produced %d hashes, want 0", len(got))
	}
}

// ---------------------------------------------------------------- prefix routing

func TestPrefixAwareSendsAConversationBackToTheSameReplica(t *testing.T) {
	// The agent-traffic case: a long shared system prompt resent every turn.
	reps := []ReplicaState{rep("a", true, 0, 0), rep("b", true, 0, 0)}
	r := NewPrefixAware(0, 0)
	sys := strings.Repeat("system prompt and tool definitions. ", 10)

	first := Request{Model: "m", PrefixBlocks: BlockHashes(sys+"turn 1", DefaultBlockSize, 32)}
	home := pick(t, r, first, reps)
	r.Observe(first, home)

	for turn := 2; turn < 12; turn++ {
		req := Request{Model: "m", PrefixBlocks: BlockHashes(fmt.Sprintf("%sturn %d", sys, turn), DefaultBlockSize, 32)}
		got := pick(t, r, req, reps)
		if got != home {
			t.Fatalf("turn %d went to %s, want %s", turn, got, home)
		}
		r.Observe(req, got)
	}
}

func TestPrefixAwareWithoutLoadPenaltyCreatesAHotSpot(t *testing.T) {
	// This is the failure mode the +load variant exists to fix, so it is pinned as a
	// test rather than left as a claim in the README.
	reps := []ReplicaState{rep("a", true, 0, 0), rep("b", true, 0, 0)}
	r := NewPrefixAware(0, 0)
	sys := strings.Repeat("shared system prompt. ", 20)

	seed := Request{Model: "m", PrefixBlocks: BlockHashes(sys, DefaultBlockSize, 32)}
	home := pick(t, r, seed, reps)
	r.Observe(seed, home)

	// The owning replica is now badly backed up, and pure affinity keeps feeding it.
	for i := range reps {
		if reps[i].ID == home {
			reps[i].QueueDepth = 100
		}
	}
	req := Request{Model: "m", PrefixBlocks: BlockHashes(sys+"another turn", DefaultBlockSize, 32)}
	if got := pick(t, r, req, reps); got != home {
		t.Fatalf("pure affinity should still pick the saturated owner, got %s", got)
	}
}

func TestPrefixAwarePlusLoadAbandonsASaturatedOwner(t *testing.T) {
	reps := []ReplicaState{rep("a", true, 0, 0), rep("b", true, 0, 0)}
	r := NewPrefixAware(1.0, 0)
	sys := strings.Repeat("shared system prompt. ", 20)

	seed := Request{Model: "m", PrefixBlocks: BlockHashes(sys, DefaultBlockSize, 32)}
	home := pick(t, r, seed, reps)
	r.Observe(seed, home)

	req := Request{Model: "m", PrefixBlocks: BlockHashes(sys+"another turn", DefaultBlockSize, 32)}

	// Mildly busy: affinity is worth more than a couple of queued requests.
	for i := range reps {
		if reps[i].ID == home {
			reps[i].QueueDepth = 1
		}
	}
	if got := pick(t, r, req, reps); got != home {
		t.Fatalf("a mild backlog should not break affinity, went to %s", got)
	}

	// Saturated: the penalty must now exceed the prefix bonus.
	for i := range reps {
		if reps[i].ID == home {
			reps[i].QueueDepth = 100
		}
	}
	if got := pick(t, r, req, reps); got == home {
		t.Fatal("load penalty failed to break affinity to a saturated replica")
	}
}

func TestPrefixMatchStopsAtTheFirstGap(t *testing.T) {
	// A KV hit needs an unbroken prefix. Blocks past a miss are worth nothing, and
	// counting them would overstate affinity and send requests to a replica that
	// would have to recompute the prefill anyway.
	r := NewPrefixAware(0, 0)
	r.owner = map[uint64]prefixEntry{1: {"a", 1}, 2: {"b", 2}, 3: {"a", 3}}

	if got := r.matchLen([]uint64{1, 2, 3}, "a"); got != 1 {
		t.Fatalf("matchLen = %d, want 1 (block 2 belongs to b)", got)
	}
	if got := r.matchLen([]uint64{1, 3}, "a"); got != 2 {
		t.Fatalf("matchLen = %d, want 2", got)
	}
}

func TestPrefixTableIsBounded(t *testing.T) {
	// Unique prompts must not grow the affinity table without limit.
	r := NewPrefixAware(0, 200)
	for i := 0; i < 5000; i++ {
		req := Request{Model: "m", PrefixBlocks: BlockHashes(
			fmt.Sprintf("%d-%s", i, strings.Repeat("x", DefaultBlockSize*2)), DefaultBlockSize, 4)}
		r.Observe(req, "a")
	}
	if len(r.owner) > 200 {
		t.Fatalf("affinity table grew to %d entries, cap is 200", len(r.owner))
	}
}

func TestPrefixAwareFallsBackCleanlyWithNoHistory(t *testing.T) {
	// A cold table must still route, not error.
	reps := []ReplicaState{rep("a", true, 7, 0), rep("b", true, 0, 0)}
	if got := pick(t, NewPrefixAware(1.0, 0), Request{Model: "m",
		PrefixBlocks: BlockHashes(strings.Repeat("q", 500), DefaultBlockSize, 32)}, reps); got != "b" {
		t.Fatalf("with no affinity data it should fall back to the lighter replica, got %s", got)
	}
}

func TestRouterChoiceIsDeterministic(t *testing.T) {
	// Replica slices arrive in whatever order the gateway's map produced. A router
	// that does not normalise that returns different replicas for identical input,
	// which would make experiment 5 unreproducible.
	a := []ReplicaState{rep("a", true, 1, 1), rep("b", true, 1, 1), rep("c", true, 1, 1)}
	b := []ReplicaState{rep("c", true, 1, 1), rep("a", true, 1, 1), rep("b", true, 1, 1)}

	for _, name := range []string{"least-loaded", "prefix-aware", "prefix-aware+load"} {
		r1, _ := NewRouter(name)
		r2, _ := NewRouter(name)
		req := Request{Model: "m", PrefixBlocks: BlockHashes(strings.Repeat("p", 400), DefaultBlockSize, 32)}
		if pick(t, r1, req, a) != pick(t, r2, req, b) {
			t.Fatalf("%s depends on input slice order", name)
		}
	}
}
