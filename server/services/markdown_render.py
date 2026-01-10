from __future__ import annotations

import bleach
import markdown as md


_ALLOWED_TAGS = [
    "a",
    "p",
    "br",
    "hr",
    "h1",
    "h2",
    "h3",
    "strong",
    "em",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
    "blockquote",
]

_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "code": ["class"],
}


def render_markdown_safe(markdown_text: str) -> str:
    html = md.markdown(
        markdown_text,
        extensions=["fenced_code", "tables"],
        output_format="html5",
    )
    cleaned = bleach.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, strip=True)
    linked = bleach.linkify(cleaned)
    return linked

