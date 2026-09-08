"""Self-contained HTML rendering of the research memo.

Figures are inlined as base64 so the file is a single artefact that survives
being emailed, attached to a ticket, or opened from a zip six months later.
Theme-aware: the palette is defined once as custom properties and redefined for
dark, under both the OS media query and an explicit theme attribute.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from etflab.report.memo import (
    Callout,
    CodeBlock,
    Figure,
    KeyValues,
    Memo,
    Paragraph,
    Table,
    build_memo,
    format_frame,
)

STYLE = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --surface-2: #f4f3f0;
  --surface-3: #ecebe7;
  --ink: #0b0b0b;
  --ink-soft: #52514e;
  --ink-mute: #7d7c77;
  --rule: #e0dfda;
  --blue: #2a78d6;
  --orange: #eb6834;
  --red: #e34948;
  --green: #008300;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #232322; --surface-3: #2c2c2a;
    --ink: #ffffff; --ink-soft: #c3c2b7; --ink-mute: #96958d; --rule: #383835;
    --blue: #3987e5; --orange: #d95926; --red: #e66767; --green: #4caf50;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #232322; --surface-3: #2c2c2a;
  --ink: #ffffff; --ink-soft: #c3c2b7; --ink-mute: #96958d; --rule: #383835;
  --blue: #3987e5; --orange: #d95926; --red: #e66767; --green: #4caf50;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 62rem; margin: 0 auto; padding: 3.5rem 1.5rem 6rem; }
header { border-bottom: 1px solid var(--rule); padding-bottom: 1.75rem; margin-bottom: 2.5rem; }
h1 { font-size: clamp(1.7rem, 3.6vw, 2.4rem); line-height: 1.18; margin: 0 0 .6rem; letter-spacing: -0.02em; }
.subtitle { font-size: 1.05rem; color: var(--ink-soft); margin: 0 0 1.2rem; }
.meta { display: flex; flex-wrap: wrap; gap: .45rem; }
.chip {
  font: 12px/1 var(--mono); background: var(--surface-2); color: var(--ink-soft);
  border: 1px solid var(--rule); border-radius: 999px; padding: .38rem .7rem;
}
nav { margin: 0 0 3rem; }
nav ol { list-style: none; margin: 0; padding: 0; display: grid; gap: .1rem; counter-reset: s; }
nav a {
  display: block; padding: .5rem .75rem; border-radius: 8px; text-decoration: none;
  color: var(--ink-soft); border-left: 2px solid transparent;
}
nav a:hover { background: var(--surface-2); color: var(--ink); border-left-color: var(--blue); }
nav a::before { counter-increment: s; content: counter(s) ". "; color: var(--ink-mute); font: 12px var(--mono); }
section { margin: 0 0 3.25rem; scroll-margin-top: 1.5rem; }
h2 {
  font-size: 1.32rem; letter-spacing: -0.01em; margin: 0 0 1.1rem;
  padding-bottom: .55rem; border-bottom: 1px solid var(--rule);
}
p { margin: 0 0 1.1rem; color: var(--ink); }
code { font: 13px var(--mono); background: var(--surface-2); padding: .12em .38em; border-radius: 4px; }
pre {
  background: var(--surface-2); border: 1px solid var(--rule); border-radius: 10px;
  padding: 1rem 1.1rem; overflow-x: auto; margin: 0 0 1.3rem;
}
pre code { background: none; padding: 0; font-size: 12.5px; line-height: 1.62; }
.callout {
  border: 1px solid var(--rule); border-left: 3px solid var(--ink-mute);
  background: var(--surface-2); border-radius: 8px; padding: 1rem 1.2rem; margin: 0 0 1.3rem;
}
.callout.finding { border-left-color: var(--blue); }
.callout.caution { border-left-color: var(--orange); }
.callout.method  { border-left-color: var(--ink-mute); }
.callout .label {
  font: 600 11px/1 var(--mono); letter-spacing: .09em; text-transform: uppercase;
  color: var(--ink-mute); display: block; margin-bottom: .5rem;
}
.callout .title { font-weight: 650; margin-bottom: .4rem; }
.callout p:last-child { margin-bottom: 0; }
.callout ul { margin: .5rem 0 0; padding-left: 1.15rem; }
.callout li { margin-bottom: .45rem; }
.table-title { font-weight: 650; margin: 1.6rem 0 .6rem; }
.scroll { overflow-x: auto; margin: 0 0 .8rem; border: 1px solid var(--rule); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: .55rem .85rem; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; font-weight: 500; }
thead th {
  background: var(--surface-2); font: 600 12px/1.4 var(--mono); color: var(--ink-soft);
  text-transform: uppercase; letter-spacing: .04em; position: sticky; top: 0;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: var(--surface-2); }
td { font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: 12.5px; }
td:first-child { font-family: inherit; font-size: 13.5px; }
.note { font-size: 13px; color: var(--ink-soft); margin: 0 0 1.4rem; }
figure { margin: 0 0 1.8rem; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--rule); border-radius: 10px; }
figcaption { font-size: 13px; color: var(--ink-soft); margin-top: .55rem; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: .5rem 1.4rem; margin: 0 0 1.4rem; }
.kv dt { color: var(--ink-mute); font-size: 13px; }
.kv dd { margin: 0; font: 13px var(--mono); word-break: break-word; }
footer { border-top: 1px solid var(--rule); padding-top: 1.4rem; color: var(--ink-mute); font-size: 13px; }
@media (max-width: 640px) { .wrap { padding: 2rem 1rem 4rem; } .kv { grid-template-columns: 1fr; gap: .15rem 0; } }
"""


def _escape(text: str) -> str:
    return html.escape(str(text), quote=False)


def _inline_image(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _render_table(table: Table) -> str:
    frame = format_frame(table)
    index_label = table.index_label or (frame.index.name or "")
    head = "".join(f"<th>{_escape(c)}</th>" for c in [index_label, *frame.columns])
    rows = "".join(
        "<tr>" + "".join(f"<td>{_escape(v)}</td>" for v in [index, *row]) + "</tr>" for index, row in frame.iterrows()
    )
    note = f'<p class="note">{_escape(table.note)}</p>' if table.note else ""
    return (
        f'<p class="table-title">{_escape(table.title)}</p>'
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>{note}'
    )


def _render_callout(block: Callout) -> str:
    lines = block.text.split("\n")
    if all(line.strip().startswith("-") for line in lines if line.strip()):
        items = "".join(f"<li>{_escape(line.strip()[1:].strip())}</li>" for line in lines if line.strip())
        body = f"<ul>{items}</ul>"
    else:
        body = "".join(f"<p>{_escape(line)}</p>" for line in lines if line.strip())
    return (
        f'<div class="callout {_escape(block.kind)}">'
        f'<span class="label">{_escape(block.kind)}</span>'
        f'<div class="title">{_escape(block.title)}</div>{body}</div>'
    )


def render_html(study: Any, manifest: Any, figure_dir: str | Path = "figures") -> str:
    """Render the memo as one self-contained HTML document."""
    memo: Memo = build_memo(study, manifest)
    figures = Path(figure_dir)
    body: list[str] = []

    nav_items = "".join(
        f'<li><a href="#{sec.anchor or sec.title.lower().replace(" ", "-")}">{_escape(sec.title)}</a></li>'
        for sec in memo.sections
    )
    body.append(f"<nav><ol>{nav_items}</ol></nav>")

    for section in memo.sections:
        anchor = section.anchor or section.title.lower().replace(" ", "-")
        chunks = [f'<section id="{anchor}"><h2>{_escape(section.title)}</h2>']
        for block in section.blocks:
            if isinstance(block, Paragraph):
                chunks.append(f"<p>{_markup(block.text)}</p>")
            elif isinstance(block, Callout):
                chunks.append(_render_callout(block))
            elif isinstance(block, Table):
                chunks.append(_render_table(block))
            elif isinstance(block, Figure):
                encoded = _inline_image(figures / f"{block.name}.png")
                if encoded:
                    chunks.append(
                        f'<figure><img alt="{_escape(block.caption)}" src="data:image/png;base64,{encoded}">'
                        f"<figcaption>{_escape(block.caption)}</figcaption></figure>"
                    )
            elif isinstance(block, KeyValues):
                items = "".join(f"<dt>{_escape(k)}</dt><dd>{_escape(v)}</dd>" for k, v in block.items)
                chunks.append(f'<p class="table-title">{_escape(block.title)}</p><dl class="kv">{items}</dl>')
            elif isinstance(block, CodeBlock):
                chunks.append(f"<pre><code>{_escape(block.text)}</code></pre>")
        chunks.append("</section>")
        body.append("".join(chunks))

    chips = "".join(f'<span class="chip">{_escape(k)} {_escape(v)}</span>' for k, v in memo.meta.items())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(memo.title)}</title>
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>{_escape(memo.title)}</h1>
<p class="subtitle">{_escape(memo.subtitle)}</p>
<div class="meta">{chips}</div>
</header>
{"".join(body)}
<footer>Generated by <code>etf-lab study</code>. Every figure and table on this page is reproducible from the
command in the summary.</footer>
</div>
</body>
</html>
"""


def _markup(text: str) -> str:
    """Minimal inline markup: `code` and **bold**, escaped everywhere else."""
    escaped = _escape(text)
    parts = escaped.split("`")
    escaped = "".join(p if i % 2 == 0 else f"<code>{p}</code>" for i, p in enumerate(parts))
    parts = escaped.split("**")
    return "".join(p if i % 2 == 0 else f"<strong>{p}</strong>" for i, p in enumerate(parts))
