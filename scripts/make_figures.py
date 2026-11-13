#!/usr/bin/env python
"""Generate the README's figures from the recorded result files.

Every figure here is drawn from docs/results/*.json and *.csv, so a figure cannot
drift away from the measurement it claims to show. Re-run after any experiment:

    python scripts/make_figures.py

Output is SVG rather than PNG: it stays sharp on any display, it is a few kilobytes
instead of a few hundred, and it diffs as text so a changed figure is reviewable.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "docs" / "results"
IMG = REPO / "docs" / "img"

# A single palette, used by every figure so they read as one set.
INK = "#0f172a"
MUTED = "#64748b"
LINE = "#cbd5e1"
HOT = "#ea580c"
WARM = "#f59e0b"
COOL = "#3b82f6"
GREEN = "#059669"

DARK_CSS = """
      @media (prefers-color-scheme: dark) {
        .bg   { fill:#0b1220; }
        .ink  { fill:#f1f5f9; }
        .mut  { fill:#94a3b8; }
        .grid { stroke:#334155; }
        .axis { stroke:#475569; }
      }"""


def head(w: int, h: int, title: str, desc: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif">
  <title>{title}</title>
  <desc>{desc}</desc>
  <style>
    .bg{{fill:#ffffff}} .ink{{fill:{INK}}} .mut{{fill:{MUTED}}}
    .grid{{stroke:{LINE};stroke-width:1;stroke-dasharray:3 4}}
    .axis{{stroke:{LINE};stroke-width:1.5}}
    .t{{font-size:19px;font-weight:700}} .s{{font-size:12.5px}}
    .v{{font-size:13px;font-weight:700}} .ax{{font-size:12px}}
    .cap{{font-size:11.5px}}{DARK_CSS}
  </style>
  <rect class="bg" width="{w}" height="{h}"/>
"""


def sleep_vs_cold() -> None:
    """Wake cost against cold start, per model, on a log scale."""
    src = RESULTS / "exp2_sleep_vs_cold.csv"
    if not src.exists():
        print("  skip sleep_vs_cold: no exp2 results")
        return
    with src.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return

    W, H = 900, 420
    L, R, T, B = 150, 40, 92, 86
    pw = W - L - R

    import math
    # Log scale: the whole point is a two-order-of-magnitude gap, which a linear
    # axis would render as three invisible bars next to one tall one.
    lo, hi = 0.3, 60.0
    def x(v: float) -> float:
        v = max(v, lo)
        return L + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    out = [head(W, H, "Wake versus cold start",
                "Waking a model from CPU RAM or disk takes under two seconds; loading it "
                "from scratch takes thirty-five.")]
    out.append(f'<text class="t ink" x="{L-110}" y="34">Bringing a model back: wake vs cold start</text>')
    out.append(f'<text class="s mut" x="{L-110}" y="56">RTX 3070 Laptop (8 GB), vLLM 0.32.0, eager mode. Log scale — each gridline is 10x.</text>')

    for gv in (0.3, 1, 3, 10, 30):
        gx = x(gv)
        out.append(f'<line class="grid" x1="{gx:.1f}" y1="{T-10}" x2="{gx:.1f}" y2="{H-B+6}"/>')
        lab = f"{gv:g}s"
        out.append(f'<text class="ax mut" x="{gx:.1f}" y="{H-B+24}" text-anchor="middle">{lab}</text>')

    bh, gap = 20, 10
    rowh = bh * 3 + gap * 2 + 26
    y = T
    series = [("wake from CPU RAM", "wake_l1_s", COOL),
              ("wake from disk", "wake_l2_s", WARM),
              ("cold start", "cold_start_s", MUTED)]

    for r in rows:
        out.append(f'<text class="v ink" x="16" y="{y+14}">{r["model"]}</text>')
        out.append(f'<text class="cap mut" x="16" y="{y+32}">{r["vram_mib"]} MiB resident</text>')
        yy = y
        for label, key, colour in series:
            v = float(r[key])
            bw = max(x(v) - L, 2)
            out.append(f'<rect x="{L}" y="{yy}" width="{bw:.1f}" height="{bh}" rx="4" fill="{colour}"/>')
            out.append(f'<text class="cap ink" x="{L+bw+8:.1f}" y="{yy+14}">{v:g}s &#183; {label}</text>')
            yy += bh + gap
        y += rowh

    out.append(f'<line class="axis" x1="{L}" y1="{T-10}" x2="{L}" y2="{H-B+6}"/>')
    speed = rows[0]
    out.append(f'<text class="s ink" x="{L}" y="{H-32}">'
               f'<tspan font-weight="700">{float(speed["l1_speedup_vs_cold"]):g}x faster</tspan>'
               f' to wake {speed["model"]} from CPU RAM than to load it from scratch.</text>')
    out.append(f'<text class="cap mut" x="{L}" y="{H-14}">'
               f'Cold start is roughly flat across model sizes because it is dominated by process spawn, '
               f'CUDA init and profiling — not by reading weights.</text>')
    out.append("</svg>")
    (IMG / "results-wake-vs-cold.svg").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("  wrote results-wake-vs-cold.svg")


def routing() -> None:
    """How evenly each router spread work across replicas, and what it cost."""
    files = sorted(RESULTS.glob("raw/exp5_*.csv"))
    if not files:
        print("  skip routing: no exp5 results")
        return

    arms = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        ok = [r for r in rows if r["status"] == "200"]
        if not ok:
            continue
        ttfts = sorted(float(r["ttft_ms"]) for r in ok if float(r["ttft_ms"]) > 0)
        if not ttfts:
            continue
        def pct(q: float) -> float:
            return ttfts[max(0, min(len(ttfts) - 1, int(round(q * (len(ttfts) - 1)))))]
        spread: dict[str, int] = {}
        for r in ok:
            spread[r["replica"]] = spread.get(r["replica"], 0) + 1
        arms.append({
            "name": f.stem.replace("exp5_", ""),
            "n": len(ok),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "spread": sorted(spread.values(), reverse=True),
        })
    if not arms:
        return

    W = 900
    H = 150 + len(arms) * 74
    L = 190
    pw = W - L - 220
    worst = max(a["p50"] for a in arms) or 1.0

    out = [head(W, H, "Router comparison",
                "Median time to first token and how evenly each router spread work "
                "across two replicas of one model.")]
    out.append(f'<text class="t ink" x="24" y="34">Routing: median TTFT and replica balance</text>')
    out.append(f'<text class="s mut" x="24" y="56">Agent-style traffic — a long system prompt and tool definitions resent every turn. '
               f'Two replicas of one model.</text>')

    y = 92
    for a in arms:
        out.append(f'<text class="v ink" x="24" y="{y+15}">{a["name"]}</text>')
        out.append(f'<text class="cap mut" x="24" y="{y+33}">{a["n"]} requests</text>')
        bw = max(pw * a["p50"] / worst, 3)
        colour = GREEN if a["p50"] <= worst * 0.4 else (WARM if a["p50"] <= worst * 0.8 else HOT)
        out.append(f'<rect x="{L}" y="{y}" width="{bw:.1f}" height="24" rx="5" fill="{colour}"/>')
        out.append(f'<text class="v ink" x="{L+bw+10:.1f}" y="{y+17}">{a["p50"]:.0f} ms p50</text>')
        bal = " / ".join(str(c) for c in a["spread"])
        out.append(f'<text class="cap mut" x="{L}" y="{y+44}">p95 {a["p95"]:.0f} ms &#183; spread across replicas: {bal}</text>')
        y += 74

    out.append(f'<text class="cap mut" x="24" y="{H-20}">An even spread is not automatically better: affinity '
               f'concentrates a conversation on one replica on purpose, to reuse its KV cache.</text>')
    out.append("</svg>")
    (IMG / "results-routing.svg").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("  wrote results-routing.svg")


def main() -> int:
    IMG.mkdir(parents=True, exist_ok=True)
    print("generating figures from recorded results")
    sleep_vs_cold()
    routing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
