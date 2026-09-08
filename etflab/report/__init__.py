"""Rendering: terminal tables, figures, and the written research memo."""

from etflab.report.console import render_study
from etflab.report.markdown import render_markdown

__all__ = ["make_figures", "render_html", "render_markdown", "render_study"]


def __getattr__(name: str):
    if name == "render_html":
        from etflab.report.html import render_html

        return render_html
    if name == "make_figures":
        from etflab.report.figures import make_figures

        return make_figures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
