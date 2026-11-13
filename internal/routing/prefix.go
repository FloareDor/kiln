package routing

import (
	"encoding/binary"
	"hash/fnv"
)

// DefaultBlockSize is how many characters of prompt go into one hashed block.
//
// vLLM's own prefix cache works in blocks of 16 tokens. The gateway does not tokenise
// — it must not, since tokenising every prompt in Go would mean shipping a tokeniser
// per model and would add latency to the hot path — so it approximates using
// characters. At roughly four characters per token, 64 characters is about one vLLM
// block.
//
// The approximation is safe in the direction that matters. Character blocks are
// coarser than token blocks, so a character-block match may slightly over- or
// under-estimate the true token prefix. It never causes a wrong answer, only a
// cache miss or a redundant affinity, and the measured prefix hit ratio in the
// experiment is read from vLLM's own counters rather than from this estimate.
const DefaultBlockSize = 64

// BlockHashes hashes the prompt into cumulative leading blocks.
//
// Each hash covers the prompt *from the start* up to the end of its block, not the
// block alone. That is what makes a prefix match mean what it should: two prompts
// share block k only if their first k blocks are byte-identical. Hashing blocks
// independently would match two prompts with the same second block but different
// first blocks, which share no KV cache at all.
func BlockHashes(prompt string, blockSize, maxBlocks int) []uint64 {
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	if maxBlocks <= 0 {
		maxBlocks = 64
	}
	n := len(prompt) / blockSize
	if n > maxBlocks {
		n = maxBlocks
	}
	out := make([]uint64, 0, n)

	h := fnv.New64a()
	for i := 0; i < n; i++ {
		// Feed only the new block each iteration and snapshot: the running hash is
		// already cumulative, so this is O(len(prompt)) overall rather than O(n^2).
		_, _ = h.Write([]byte(prompt[i*blockSize : (i+1)*blockSize]))
		out = append(out, h.Sum64())
	}
	return out
}

// HashKey renders a block hash as a fixed-width key, used for metrics labels and the
// decision log.
func HashKey(h uint64) [8]byte {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], h)
	return b
}
