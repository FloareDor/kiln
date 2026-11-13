package api

import (
	"crypto/subtle"

	"github.com/FloareDor/kiln/internal/routing"
)

// subtleCompare is a constant-time string comparison. A short-circuiting compare on a
// shared admin token leaks how many leading characters matched, which with an online
// oracle is enough to recover it one character at a time.
func subtleCompare(a, b string) int {
	return subtle.ConstantTimeCompare([]byte(a), []byte(b))
}

func replicaID(s string) routing.ReplicaID { return routing.ReplicaID(s) }
