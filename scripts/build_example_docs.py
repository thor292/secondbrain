"""Build the example PDF and Word documents from their Markdown sources.

Real companies mostly have PDF and Word files, so part of the example data
ships in those formats. Sources live in examples/<company>/source/ and are
named <brain>__<sensitivity>__<Name>.<pdf|docx>.md; outputs go to
examples/<company>/docs/<brain>/<sensitivity>/<Name>.<pdf|docx>.

    uv run python scripts/build_example_docs.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

EXAMPLES = Path(__file__).parent.parent / "examples"

# The standard PDF fonts have no rupee sign (₹), so embed a TrueType font that has one.
# Without any of these, the PDFs fall back to Helvetica and "Rs." instead of "₹".
UNICODE_FONTS = [
    "/System/Library/Fonts/Geneva.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian, Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Fedora
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def pdf_font() -> str | None:
    """Register and return the name of a font with the rupee sign, if one is installed."""
    for path in UNICODE_FONTS:
        if Path(path).exists():
            font = TTFont("Unicode", path)
            if ord("₹") in font.face.charToGlyph:
                pdfmetrics.registerFont(font)
                return "Unicode"
    return None


def parse(markdown: str) -> list[tuple[str, object]]:
    """Blocks of ("title" | "heading" | "paragraph", text) or ("table", rows)."""
    blocks, table = [], []
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("|"):
            table.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if table:
            blocks.append(("table", table))
            table = []
        if line.startswith("## "):
            blocks.append(("heading", line[3:]))
        elif line.startswith("# "):
            blocks.append(("title", line[2:]))
        elif line:
            blocks.append(("paragraph", line))
    if table:
        blocks.append(("table", table))
    return blocks


def write_docx(blocks, out: Path) -> None:
    doc = Document()
    for kind, value in blocks:
        if kind == "title":
            doc.add_heading(value, 0)
        elif kind == "heading":
            doc.add_heading(value, 1)
        elif kind == "paragraph":
            doc.add_paragraph(value)
        else:
            table = doc.add_table(rows=len(value), cols=len(value[0]))
            table.style = "Light Grid Accent 1"
            for r, row in enumerate(value):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
    doc.save(out)


def write_pdf(blocks, out: Path, font: str | None) -> None:
    styles = getSampleStyleSheet()
    if font:
        for name in ("Title", "Heading2", "BodyText"):
            styles[name].fontName = font

    def text(value: str) -> str:
        return escape(value if font else value.replace("₹", "Rs. "))

    width = A4[0] - 2 * 72  # page width inside the default margins
    story = []
    for kind, value in blocks:
        if kind == "title":
            story += [Paragraph(text(value), styles["Title"]), Spacer(1, 6)]
        elif kind == "heading":
            story.append(Paragraph(text(value), styles["Heading2"]))
        elif kind == "paragraph":
            story.append(Paragraph(text(value), styles["BodyText"]))
        else:
            rows = [[Paragraph(text(cell), styles["BodyText"]) for cell in row] for row in value]
            table = Table(rows, colWidths=[width / len(value[0])] * len(value[0]))
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [table, Spacer(1, 6)]
    SimpleDocTemplate(str(out), pagesize=A4, title=blocks[0][1] if blocks else "").build(story)


def main() -> None:
    font = pdf_font()
    if font is None:
        print("no font with the rupee sign found; PDFs will say Rs. instead of ₹")
    for source in sorted(EXAMPLES.glob("*/source/*.md")):
        brain, sensitivity, filename = source.name.removesuffix(".md").split("__")
        out = source.parent.parent / "docs" / brain / sensitivity / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        blocks = parse(source.read_text())
        if out.suffix == ".pdf":
            write_pdf(blocks, out, font)
        else:
            write_docx(blocks, out)
        print(f"built {out.relative_to(EXAMPLES)}")


if __name__ == "__main__":
    main()
