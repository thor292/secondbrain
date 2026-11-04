"""Turning company documents into memory-sized pieces.

Supports Markdown, plain text, PDF and Word (.docx). Text is split along the
document's own sections, and every chunk starts with "Title › Section" so it
still makes sense on its own when it is retrieved later.
"""

import io
import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
CHUNK_CHARS = 1200
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


class UnsupportedDocument(ValueError):
    pass


def extract_text(name: str, data: bytes) -> str:
    """Plain text of a document, with headings marked Markdown-style (``# Title``)."""
    suffix = Path(name).suffix.lower()
    if suffix in {".md", ".txt"}:
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        return _pdf_text(data)
    if suffix == ".docx":
        return _docx_text(data)
    raise UnsupportedDocument(f"{name}: supported types are {', '.join(sorted(SUPPORTED_SUFFIXES))}")


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    lines = []
    for page in PdfReader(io.BytesIO(data)).pages:
        lines += (page.extract_text() or "").splitlines()
        lines.append("")
    # PDFs carry no structure, so rebuild it: a short line without closing
    # punctuation, followed by more text, is treated as a heading; other
    # consecutive lines are joined back into paragraphs.
    out, paragraph = [], []

    def flush() -> None:
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            flush()
            continue
        nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        looks_like_heading = (
            len(line) <= 60 and not line.endswith((".", ",", ";", ":")) and nxt
            and not line[0].islower() and not paragraph
        )
        if looks_like_heading:
            flush()
            # The first heading in the document is its title.
            out.append(f"{'#' if not out else '##'} {line}")
        else:
            paragraph.append(line)
            if line.endswith((".", "!", "?")):
                flush()  # chunk() regroups short paragraphs later
    flush()
    return "\n\n".join(out)


def _docx_text(data: bytes) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(io.BytesIO(data))
    out = []
    # Walk the body in order so tables stay where they appear in the document.
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "") if paragraph.style is not None else ""
            if style == "Title":
                out.append(f"# {text}")
            elif style.startswith("Heading"):
                level = int(style.split()[-1]) if style.split()[-1].isdigit() else 2
                out.append(f"{'#' * min(level + 1, 6)} {text}")
            elif style.startswith("List"):
                out.append(f"- {text}")
            else:
                out.append(text)
        elif tag == "tbl":
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in Table(element, document).rows]
            out.append("\n".join(rows))
    return "\n\n".join(out)


def chunk(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split text on paragraph boundaries into chunks of at most ~max_chars.

    A single paragraph longer than max_chars is split between sentences.
    """
    paragraphs = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not p:
            continue
        if len(p) <= max_chars:
            paragraphs.append(p)
            continue
        piece = ""
        for sentence in re.split(r"(?<=[.!?])\s+", p):
            if piece and len(piece) + len(sentence) + 1 > max_chars:
                paragraphs.append(piece)
                piece = sentence
            else:
                piece = f"{piece} {sentence}" if piece else sentence
        if piece:
            paragraphs.append(piece)

    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def sections(text: str) -> list[tuple[str | None, str | None, str]]:
    """Split on headings: (document title, section heading, body) per section."""
    title, heading, body, out = None, None, [], []
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            if body and "".join(body).strip():
                out.append((title, heading, "\n".join(body)))
            body = []
            level, text_ = len(match.group(1)), match.group(2).strip()
            if level == 1 and title is None:
                title, heading = text_, None
            else:
                heading = text_
        else:
            body.append(line)
    if body and "".join(body).strip():
        out.append((title, heading, "\n".join(body)))
    return out


def chunk_document(name: str, text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Chunks labeled "Title › Section", so each stands on its own when retrieved."""
    fallback_title = Path(name).stem.replace("_", " ").replace("-", " ").strip().capitalize()
    pieces = []
    for title, heading, body in sections(text):
        label = title or fallback_title
        if heading and heading != label:
            label = f"{label} › {heading}"
        for part in chunk(body, max_chars):
            pieces.append(f"{label}\n\n{part}")
    return pieces
