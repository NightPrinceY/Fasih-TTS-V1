"""Fasih Studio look and feel: theme, CSS and static HTML, built from the Fasih identity kit."""

import gradio as gr

# Identity palette (VisionIdentity/Fasih-TTS-V1 - Identity.dc.html)
INK = "#000000"       # page, pure black like the hero banner and mark
SURFACE = "#1a1614"   # cards
RULE = "#2a2420"      # borders, mark outline
CREAM = "#f5ead8"     # primary text
SAND = "#cbbda4"      # secondary text
EMBER = "#d67f48"     # accent, mark bars
EMBER_DEEP = "#c67139"
CLAY = "#8c491a"      # boundary panel

_ember = gr.themes.Color(
    c50="#fbf1ea", c100="#f5dccb", c200="#ecbd9d", c300="#e29d70", c400="#d67f48",
    c500="#c67139", c600="#a85b2b", c700="#8c491a", c800="#6b3814", c900="#4a270e", c950="#2e1809",
)
_warm = gr.themes.Color(
    c50="#f5ead8", c100="#e0d2bb", c200="#cbbda4", c300="#a8997f", c400="#7d705e",
    c500="#5a4f43", c600="#3d342d", c700="#2a2420", c800="#1f1a17", c900="#1a1614", c950="#121010",
)


_BASE = gr.themes.Base(
    primary_hue=_ember,
    secondary_hue=_ember,
    neutral_hue=_warm,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Figtree"), "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "monospace"],
)


def _both(**kw):
    """Same value in light and dark mode (the studio is dark only), where a dark variant exists."""
    out = dict(kw)
    out.update({f"{k}_dark": v for k, v in kw.items() if hasattr(_BASE, f"{k}_dark")})
    return out


THEME = _BASE.set(**_both(
    body_background_fill=INK,
    body_text_color=CREAM,
    body_text_color_subdued=SAND,
    background_fill_primary=SURFACE,
    background_fill_secondary=INK,
    block_background_fill=SURFACE,
    block_border_color=RULE,
    block_border_width="1px",
    block_label_background_fill=SURFACE,
    block_label_text_color=SAND,
    block_title_text_color=SAND,
    border_color_primary=RULE,
    border_color_accent=EMBER,
    color_accent=EMBER,
    color_accent_soft="#2e1f17",
    input_background_fill=INK,
    input_border_color=RULE,
    input_border_color_focus=EMBER,
    button_primary_background_fill=EMBER,
    button_primary_background_fill_hover=EMBER_DEEP,
    button_primary_text_color=INK,
    button_secondary_background_fill=INK,
    button_secondary_background_fill_hover="#241e1b",
    button_secondary_text_color=CREAM,
    button_secondary_border_color=RULE,
    slider_color=EMBER,
    checkbox_background_color_selected=EMBER,
    checkbox_border_color=RULE,
    link_text_color=EMBER,
    shadow_drop="none",
    panel_background_fill=SURFACE,
    panel_border_color=RULE,
))

HEAD = """
<meta name="theme-color" content="#000000">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,700..900,100&family=Figtree:wght@400;500;600;700&family=Reem+Kufi:wght@500;600;700&family=Noto+Naskh+Arabic:wght@400;500;600&display=swap" rel="stylesheet">
"""

# Gradio switches to dark mode with a class on <body>; the studio always runs dark.
FORCE_DARK = "() => { document.body.classList.add('dark'); }"

MARK_SVG = (
    '<svg viewBox="0 0 100 100" aria-hidden="true">'
    f'<rect x="1" y="1" width="98" height="98" rx="26" fill="#000" stroke="{RULE}" stroke-width="2"/>'
    f'<rect class="b1" x="24" y="37" width="12" height="26" rx="6" fill="{EMBER}"/>'
    f'<rect class="b2" x="44" y="26" width="12" height="48" rx="6" fill="{EMBER}"/>'
    f'<rect class="b3" x="64" y="35" width="12" height="30" rx="6" fill="{EMBER}"/>'
    "</svg>"
)

HERO = f"""
<header id="hero">
  <div class="mark">{MARK_SVG}</div>
  <div class="titles">
    <h1>Fasih-TTS-V1 <span class="ar" lang="ar">فصيح</span></h1>
    <p class="tagline">The voice <span lang="ar">مسلم</span> answers in.</p>
    <p class="lede" lang="ar" dir="rtl">اكتب بالعربية الفصحى، مشكولة أو بلا تشكيل، واسمعها بصوت مذيع محترف.</p>
  </div>
  <nav class="links" aria-label="Fasih resources">
    <a href="https://huggingface.co/NightPrince/Fasih-TTS-V1">Model</a>
    <a href="https://arxiv.org/abs/2609.31511">Paper</a>
    <a href="https://github.com/NightPrinceY/Fasih-TTS-V1">Code</a>
    <a href="https://huggingface.co/spaces/Navid-AI/Arabic-TTS-Arena">Arena</a>
  </nav>
</header>
"""

READ_EMPTY = """
<div class="read empty" dir="rtl" lang="ar">
  <p class="read-label">ما قرأه فصيح <span lang="en">What Fasih read</span></p>
  <p class="read-hint">بعد التوليد يظهر هنا النص كما نطقه فصيح: بالتشكيل الكامل، والأرقام مكتوبة بالحروف.</p>
</div>
"""

STATS_EMPTY = '<div class="stats idle" dir="rtl" lang="ar">جاهز. اكتب نصًا واضغط «انطق».</div>'


def read_html(chunks: list[str]) -> str:
    lines = "".join(f"<p class='read-text'>{c}</p>" for c in chunks)
    return (
        "<div class='read' dir='rtl' lang='ar'>"
        "<p class='read-label'>ما قرأه فصيح <span lang='en'>What Fasih read</span></p>"
        f"{lines}</div>"
    )


def stats_html(audio_s: float, gen_s: float, n_chunks: int) -> str:
    def item(value: str, ar: str, en: str) -> str:
        return f"<div class='stat'><b>{value}</b><span>{ar}<i lang='en'>{en}</i></span></div>"
    return (
        "<div class='stats' dir='rtl' lang='ar'>"
        + item(f"{audio_s:.1f} ث", "مدة الصوت", "audio")
        + item(f"{gen_s:.1f} ث", "زمن التوليد", "generation")
        + item(str(n_chunks), "مقاطع", "chunks")
        + "</div>"
    )


def counter_html(n: int) -> str:
    return f"<div class='counter' dir='rtl' lang='ar'>{n} حرف</div>"


COUNTER_JS = "(t) => `<div class='counter' dir='rtl' lang='ar'>${(t || '').length} حرف</div>`"


EVIDENCE = """
<section id="evidence" aria-labelledby="ev-title">
  <h2 id="ev-title">Why it sounds right <span lang="ar">لماذا يبدو صحيحًا</span></h2>
  <div class="facts">
    <div class="fact">
      <p class="num">1.3%</p>
      <p class="what">Character error rate when Whisper transcribes Fasih. The original human recordings score 1.8%.</p>
    </div>
    <div class="fact">
      <p class="num">0 / 24</p>
      <p class="what">Loops, skips or early cut-offs across 24 stress generations.</p>
    </div>
    <div class="fact">
      <p class="num">675 ms</p>
      <p class="what">To first streamed audio on one RTX 2080 Ti, with a real-time factor of 0.60.</p>
    </div>
  </div>

  <div class="silma">
    <h3>SILMA Arabic TTS benchmark, MSA <span>word error rate with Whisper, lower is better</span></h3>
    <ol class="bars">
      <li class="ours"><span class="name">Fasih-TTS-V1</span><span class="track"><span class="fill" style="width:29.7%"></span></span><span class="val">6.5</span></li>
      <li><span class="name">XTTS v2 (base)</span><span class="track"><span class="fill" style="width:47.0%"></span></span><span class="val">10.3</span></li>
      <li><span class="name">silma_tts</span><span class="track"><span class="fill" style="width:50.7%"></span></span><span class="val">11.1</span></li>
      <li><span class="name">chatterbox</span><span class="track"><span class="fill" style="width:58.4%"></span></span><span class="val">12.8</span></li>
      <li><span class="name">omnivoice</span><span class="track"><span class="fill" style="width:69.9%"></span></span><span class="val">15.3</span></li>
      <li><span class="name">habibi_specialized</span><span class="track"><span class="fill" style="width:100%"></span></span><span class="val">21.9</span></li>
    </ol>
    <p class="note">Fasih also has the lowest error with NVIDIA NeMo as the judge (2.5, tied with base XTTS v2).
      On naturalness (UTMOS) it ranks third. Full results:
      <a href="https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark">Fasih-TTS-Benchmark</a>.</p>
  </div>
</section>
"""

BOUNDARY = """
<section id="boundary">
  <p class="ar" lang="ar" dir="rtl">لا يقرأ القرآن. للقرآن قرّاؤه.</p>
  <p class="en">It never recites. The Qur'an has its reciters.</p>
  <p class="small">Fasih reads explanatory religious and educational Fusha that a qualified person has
    written or reviewed. Do not use it for recitation, religious rulings, or to put words in anyone's mouth.</p>
</section>
"""

FOOTER = """
<footer id="foot">
  <p>Fine-tuned from Coqui XTTS v2 · Coqui Public Model License, non-commercial ·
    © 2026 Yahya Elnawasany (NightPrince)</p>
  <p>Submitted text and generated audio may be logged privately to improve the model.</p>
</footer>
"""

CSS = f"""
:root {{
  --ink:{INK}; --surface:{SURFACE}; --rule:{RULE}; --cream:{CREAM}; --sand:{SAND};
  --ember:{EMBER}; --ember-deep:{EMBER_DEEP}; --clay:{CLAY};
  --display:'Fraunces', Georgia, serif;
  --kufi:'Reem Kufi', 'Noto Naskh Arabic', serif;
  --naskh:'Noto Naskh Arabic', 'Amiri', serif;
}}
body, .gradio-container {{ background: var(--ink) !important; }}
.gradio-container {{ max-width: 1120px !important; margin: 0 auto !important; padding: 0 16px !important; }}
footer.svelte-1rjryqp, .built-with {{ opacity: .55; }}

/* hero */
#hero {{ display:grid; grid-template-columns: 88px 1fr auto; gap: 22px; align-items:center;
  padding: 36px 0 26px; border-bottom: 1px solid var(--rule); margin-bottom: 22px; }}
#hero .mark svg {{ width: 88px; height: 88px; display:block; }}
#hero h1 {{ font-family: var(--display); font-weight: 900; font-variation-settings: 'SOFT' 100, 'opsz' 144;
  font-size: clamp(2.1rem, 4.6vw, 3.4rem); line-height: 1; letter-spacing: -.01em; color: var(--cream); margin: 0; }}
[lang="ar"], .read-label, #chips button, #speak-btn {{ word-spacing: .18em; }}
#hero h1 {{ white-space: nowrap; }}
#hero h1 .ar {{ font-family: var(--kufi); font-weight: 600; color: var(--ember); font-size: .78em; margin-inline-start: .35em; }}
#hero .tagline {{ font-family: var(--display); font-weight: 800; font-size: 1.25rem; color: var(--sand); margin: 10px 0 0; }}
#hero .tagline span {{ font-family: var(--kufi); color: var(--cream); }}
#hero .lede {{ font-family: var(--naskh); color: var(--sand); font-size: 1.02rem; margin: 6px 0 0; width: fit-content; max-width: 100%; }}
#hero .links {{ display:flex; flex-direction: column; gap: 6px; align-self: start; padding-top: 6px; }}
#hero .links a {{ color: var(--sand); text-decoration: none; font-weight: 600; font-size: .92rem;
  border-bottom: 1px solid transparent; justify-self: end; text-align: end; }}
#hero .links a:hover, #hero .links a:focus-visible {{ color: var(--ember); border-bottom-color: var(--ember); outline: none; }}

/* studio */
#studio {{ gap: 18px; }}
#studio > .column {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 22px; padding: 18px; }}
#text-in textarea {{ font-family: var(--naskh) !important; font-size: 1.35rem !important; line-height: 2.05 !important;
  direction: rtl; text-align: right; color: var(--cream) !important; padding: 14px 16px !important; }}
.counter {{ color: var(--sand); font-size: .8rem; opacity: .8; margin-top: -4px; font-variant-numeric: tabular-nums; }}
#chips {{ gap: 8px; flex-wrap: wrap; }}
#chips button {{ font-family: var(--kufi) !important; font-weight: 500 !important; font-size: 1rem !important;
  min-width: 0 !important; flex: 0 0 auto !important; padding: 6px 16px !important; border-radius: 999px !important; }}
#speak-btn {{ font-family: var(--kufi) !important; font-weight: 700 !important; font-size: 1.45rem !important;
  min-height: 58px; border-radius: 16px !important; letter-spacing: 0; }}
#speak-btn:focus-visible {{ outline: 2px solid var(--cream); outline-offset: 3px; }}

/* output */
.stats {{ display:flex; gap: 10px; flex-wrap: wrap; font-family: var(--naskh); }}
.stats.idle {{ color: var(--sand); font-size: 1rem; padding: 6px 2px; }}
.stat {{ flex: 1 1 0; min-width: 96px; border: 1px solid var(--rule); border-radius: 14px; padding: 10px 12px; }}
.stat b {{ display:block; font-family: var(--display); font-weight: 800; font-size: 1.5rem; color: var(--cream);
  direction: ltr; text-align: right; font-variant-numeric: tabular-nums; }}
.stat span {{ display:block; color: var(--sand); font-size: .92rem; }}
.stat i {{ display:block; font-style: normal; font-family: 'Figtree', sans-serif; font-size: .75rem; opacity: .7; }}

.read {{ border-top: 1px solid var(--rule); margin-top: 6px; padding-top: 14px; }}
.read-label {{ font-family: var(--kufi); color: var(--ember); font-size: 1.05rem; margin: 0 0 6px; }}
.read-label span {{ font-family: 'Figtree', sans-serif; color: var(--sand); font-size: .78rem; margin-inline-start: 8px; }}
.read-text {{ font-family: var(--naskh); font-size: 1.6rem; line-height: 2.25; color: var(--cream); margin: 0 0 4px; }}
.read-hint {{ font-family: var(--naskh); color: var(--sand); font-size: 1rem; line-height: 1.9; margin: 0; }}
.read:not(.empty) {{ animation: reveal .5s ease-out; }}
@keyframes reveal {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: none; }} }}

/* evidence */
#evidence {{ margin-top: 40px; }}
#evidence h2 {{ font-family: var(--display); font-weight: 900; font-variation-settings: 'SOFT' 100;
  font-size: 2rem; color: var(--cream); margin: 0 0 18px; }}
#evidence h2 span {{ font-family: var(--kufi); font-weight: 500; color: var(--sand); font-size: .6em; margin-inline-start: 10px; }}
.facts {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 0; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }}
.fact {{ padding: 20px 22px 22px 0; }}
.fact + .fact {{ border-inline-start: 1px solid var(--rule); padding-inline-start: 22px; }}
.fact .num {{ font-family: var(--display); font-weight: 900; font-variation-settings: 'SOFT' 100;
  font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1; color: var(--ember); margin: 0 0 10px; font-variant-numeric: tabular-nums; }}
.fact .what {{ color: var(--sand); font-size: .98rem; line-height: 1.55; margin: 0; max-width: 32ch; }}

.silma {{ margin-top: 34px; }}
.silma h3 {{ font-family: var(--display); font-weight: 800; font-size: 1.3rem; color: var(--cream); margin: 0 0 14px; }}
.silma h3 span {{ display:block; font-family: 'Figtree', sans-serif; font-weight: 500; font-size: .9rem; color: var(--sand); margin-top: 4px; }}
.bars {{ list-style: none; margin: 0; padding: 0; display:grid; gap: 10px; }}
.bars li {{ display:grid; grid-template-columns: 180px 1fr 48px; gap: 14px; align-items:center; color: var(--sand); }}
.bars .track {{ height: 12px; background: #221c19; border-radius: 999px; overflow: hidden; }}
.bars .fill {{ display:block; height: 100%; background: #4a4038; border-radius: 999px; }}
.bars .ours {{ color: var(--cream); font-weight: 700; }}
.bars .ours .fill {{ background: var(--ember); }}
.bars .val {{ text-align: right; font-variant-numeric: tabular-nums; font-family: var(--display); font-weight: 800; }}
.silma .note {{ color: var(--sand); font-size: .92rem; line-height: 1.6; margin-top: 14px; max-width: 75ch; }}

/* boundary */
#boundary {{ margin-top: 40px; background: var(--clay); border-radius: 22px; padding: 30px 30px 26px; }}
#boundary .ar {{ font-family: var(--kufi); font-weight: 600; font-size: clamp(1.7rem, 4vw, 2.5rem); color: var(--cream); margin: 0; text-align: start; }}
#boundary .en {{ font-family: var(--display); font-weight: 800; font-size: 1.35rem; color: #ffe7d6; margin: 8px 0 0; }}
#boundary .small {{ color: #f3dcc6; font-size: .95rem; line-height: 1.6; margin: 12px 0 0; max-width: 70ch; }}

#foot {{ margin: 28px 0 10px; color: var(--sand); opacity: .75; font-size: .82rem; line-height: 1.6; }}
#foot p {{ margin: 0; }}

@media (max-width: 760px) {{
  #hero {{ grid-template-columns: 64px 1fr; padding-top: 22px; align-items: start; }}
  #hero .mark svg {{ width: 64px; height: 64px; }}
  #hero h1 {{ font-size: 2.15rem; }}
  #hero h1 .ar {{ display: block; margin: 6px 0 0; font-size: .8em; }}
  #hero .links {{ grid-column: 1 / -1; flex-direction: row; flex-wrap: wrap; gap: 16px; }}
  .facts {{ grid-template-columns: 1fr; }}
  .fact, .fact + .fact {{ border-inline-start: 0; padding-inline-start: 0; }}
  .fact + .fact {{ border-top: 1px solid var(--rule); }}
  .bars li {{ grid-template-columns: 120px 1fr 40px; font-size: .9rem; }}
  #text-in textarea {{ font-size: 1.18rem !important; }}
  .read-text {{ font-size: 1.35rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ .read:not(.empty) {{ animation: none; }} }}
"""
