package nodeagent

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// GPUInfo is one device's memory situation, in bytes.
type GPUInfo struct {
	Index     int    `json:"index"`
	Name      string `json:"name"`
	TotalBytes int64 `json:"total_bytes"`
	UsedBytes  int64 `json:"used_bytes"`
	FreeBytes  int64 `json:"free_bytes"`
	// Source records how the numbers were obtained, so a surprising benchmark can be
	// traced to its measurement method rather than guessed at.
	Source string `json:"source"`
}

// HostInfo is the node's memory, which bounds how many level-1 sleepers can be kept
// warm. Host RAM, not VRAM, is the real ceiling on the host tier.
type HostInfo struct {
	TotalBytes     int64 `json:"total_bytes"`
	AvailableBytes int64 `json:"available_bytes"`
}

// GPUProbe reads device memory.
//
// It shells out to nvidia-smi rather than binding NVML through cgo. The reasons are
// practical: cgo complicates cross-compilation and the container image, the query
// runs once per scheduler tick rather than per request, and NVML's per-process
// accounting — the only thing cgo would buy — does not work under WSL2 anyway, which
// M0 established the hard way. Device-wide totals are what the budget needs, and
// nvidia-smi reports those correctly everywhere.
type GPUProbe struct {
	mu     sync.Mutex
	cached []GPUInfo
	at     time.Time
	ttl    time.Duration
}

// NewGPUProbe builds a probe that caches for ttl. nvidia-smi takes tens of
// milliseconds, which is fine per tick and not fine per request.
func NewGPUProbe(ttl time.Duration) *GPUProbe {
	if ttl <= 0 {
		ttl = time.Second
	}
	return &GPUProbe{ttl: ttl}
}

// Query returns per-device memory, cached.
func (p *GPUProbe) Query(ctx context.Context) ([]GPUInfo, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if time.Since(p.at) < p.ttl && p.cached != nil {
		return p.cached, nil
	}
	infos, err := nvidiaSMI(ctx)
	if err != nil {
		return nil, err
	}
	p.cached, p.at = infos, time.Now()
	return infos, nil
}

func nvidiaSMI(ctx context.Context) ([]GPUInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.used,memory.free",
		"--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nodeagent: nvidia-smi: %w", err)
	}
	var infos []GPUInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			continue
		}
		mib := func(s string) int64 {
			v, _ := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
			return v * 1024 * 1024
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		infos = append(infos, GPUInfo{
			Index: idx, Name: strings.TrimSpace(parts[1]),
			TotalBytes: mib(parts[2]), UsedBytes: mib(parts[3]), FreeBytes: mib(parts[4]),
			Source: "nvidia-smi",
		})
	}
	if len(infos) == 0 {
		return nil, fmt.Errorf("nodeagent: nvidia-smi returned no devices")
	}
	return infos, nil
}

// ReadHostInfo reads total and available memory from /proc/meminfo.
//
// MemAvailable rather than MemFree: free memory excludes the page cache, which the
// kernel will reclaim under pressure, and using it would massively understate how
// many level-1 sleepers fit — exactly the mistake that leads to needless demotions to
// disk.
func ReadHostInfo() (HostInfo, error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return HostInfo{}, err
	}
	defer f.Close()

	var info HostInfo
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			info.TotalBytes = kb * 1024
		case "MemAvailable:":
			info.AvailableBytes = kb * 1024
		}
	}
	return info, sc.Err()
}
