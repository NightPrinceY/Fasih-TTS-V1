"""Generate the Fasih-TTS-V1 architecture diagram (SVG). Dependency-free.

Renders a dark, modern two-lane diagram: the training pipeline and the inference runtime,
with a metrics footer. Output: assets/architecture.svg
"""

from __future__ import annotations

import html
from pathlib import Path

W, H = 1300, 780
BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"

OUT = Path("assets/architecture.svg")


def box(x, y, w, h, title, sub, fill, *, glow=False, accent="#ffffff"):
    r = 14
    s = ""
    if glow:
        s += f'<rect x="{x-4}" y="{y-4}" width="{w+8}" height="{h+8}" rx="{r+3}" fill="none" stroke="{accent}" stroke-width="2" opacity="0.55"/>'
    s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"/>'
    s += f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>'
    s += (f'<text x="{x+w/2}" y="{y+h/2-6}" text-anchor="middle" '
          f'font-size="18" font-weight="700" fill="#ffffff">{html.escape(title)}</text>')
    if sub:
        s += (f'<text x="{x+w/2}" y="{y+h/2+16}" text-anchor="middle" '
              f'font-size="12.5" fill="#cbd5e1">{html.escape(sub)}</text>')
    return s


def arrow(x1, y1, x2, y2, color="#8b949e"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="2.5" marker-end="url(#ah)"/>')


def row(items, y, h=78):
    """Place boxes evenly across the width; return svg + list of (cx, x, w)."""
    n = len(items)
    margin, gap = 55, 26
    w = (W - 2 * margin - (n - 1) * gap) / n
    svg, centers = "", []
    x = margin
    for (title, sub, fill, accent, glow) in items:
        svg += box(x, y, w, h, title, sub, fill, glow=glow, accent=accent)
        centers.append((x + w / 2, x, w, y, h))
        x += w + gap
    # arrows between consecutive boxes
    for i in range(n - 1):
        cx, xi, wi, yy, hh = centers[i]
        svg += arrow(xi + wi, y + h / 2, centers[i + 1][1], y + h / 2)
    return svg, centers


def pill(x, y, label, value, w=270):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="60" rx="12" fill="#161b22" stroke="#30363d"/>'
    s += f'<text x="{x+18}" y="{y+26}" font-size="13" fill="{MUTED}">{html.escape(label)}</text>'
    s += f'<text x="{x+18}" y="{y+48}" font-size="20" font-weight="800" fill="#3fb950">{html.escape(value)}</text>'
    return s


def main() -> int:
    TEAL, VIOLET, SLATE, BLUE, AMBER = "#0f766e", "#7c3aed", "#334155", "#1f6feb", "#b45309"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">']
    parts.append('<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="3" '
                 'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#8b949e"/>'
                 '</marker></defs>')
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    # header
    parts.append(f'<text x="55" y="58" font-size="34" font-weight="800" fill="{FG}">🕌 Fasih-TTS-V1</text>')
    parts.append(f'<text x="55" y="86" font-size="16" fill="{MUTED}">'
                 'Arabic (MSA / Fusha) professional-male Text-to-Speech — system architecture</text>')

    # ---- TRAINING lane ----
    parts.append(f'<text x="55" y="132" font-size="14" font-weight="700" fill="{MUTED}" '
                 'letter-spacing="2">TRAINING PIPELINE</text>')
    train = [
        ("Dataset", "1517 clips · MP3", SLATE, "#94a3b8", False),
        ("Validate", "QC · manifests", SLATE, "#94a3b8", False),
        ("Diacritize", "CATT · tashkeel", TEAL, "#5eead4", False),
        ("Preprocess", "24 kHz · trim", SLATE, "#94a3b8", False),
        ("Fine-tune", "XTTS v2 · FP32", BLUE, "#7ab7ff", False),
        ("Evaluate", "CER · Whisper", SLATE, "#94a3b8", False),
    ]
    svg, tcent = row(train, 150)
    parts.append(svg)

    # ---- INFERENCE lane ----
    parts.append(f'<text x="55" y="392" font-size="14" font-weight="700" fill="{MUTED}" '
                 'letter-spacing="2">INFERENCE RUNTIME</text>')
    infer = [
        ("Raw Arabic text", "even undiacritized", SLATE, "#94a3b8", False),
        ("Text Front-End", "normalize·numbers·diacritize·lexicon·chunk", TEAL, "#5eead4", False),
        ("Fasih-TTS-V1", "XTTS v2 fine-tuned", VIOLET, "#c4b5fd", True),
        ("24 kHz Speech", "mono waveform", SLATE, "#94a3b8", False),
        ("Serving", "FastAPI stream · CLI", AMBER, "#fbbf24", False),
    ]
    svg, icent = row(infer, 410, h=82)
    parts.append(svg)

    # connector: training Fine-tune -> inference Fasih model
    fx = tcent[4][0]
    mx = icent[2][0]
    parts.append(f'<path d="M{fx},{150+78} C{fx},310 {mx},300 {mx},{410}" fill="none" '
                 'stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="6 5" marker-end="url(#ah)"/>')
    parts.append(f'<text x="{(fx+mx)/2+8}" y="330" font-size="12.5" fill="#c4b5fd" '
                 'font-style="italic">produces the model</text>')

    # ---- metrics footer ----
    parts.append(f'<text x="55" y="576" font-size="14" font-weight="700" fill="{MUTED}" '
                 'letter-spacing="2">BENCHMARKS</text>')
    y = 596
    parts.append(pill(55, y, "Intelligibility (CER)", "1.3%  ≈ human 1.8%"))
    parts.append(pill(350, y, "Run-to-run variance", "0.0  (identical ×4)"))
    parts.append(pill(645, y, "Real-time factor", "RTF ~0.60"))
    parts.append(pill(940, y, "Streaming first-audio", "~675 ms", w=305))

    # footnote
    parts.append(f'<text x="55" y="{H-24}" font-size="12.5" fill="{MUTED}">'
                 'Base: Coqui XTTS v2  ·  Diacritization: CATT  ·  ASR judge: Whisper-large-v3  ·  '
                 'Single professional male voice, ~2.4 h</text>')

    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
