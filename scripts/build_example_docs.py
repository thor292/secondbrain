"""Build the example PDF and Word documents from their Markdown sources.

Real companies mostly have PDF and Word files, so part of the example data
ships in those formats. Sources live in examples/<company>/source/ and are
named <brain>__<sensitivity>__<Name>.<pdf|docx>.md; outputs go to
examples/<company>/docs/<brain>/<sensitivity>/<Name>.<pdf|docx>.

    uv run python scripts/build_example_docs.py
"""

from pathlib import Path

from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

EXAMPLES = Path(__file__).parent.parent / "examples"


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


def write_pdf(blocks, out: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for kind, value in blocks:
        if kind == "title":
            story += [Paragraph(value, styles["Title"]), Spacer(1, 6)]
        elif kind == "heading":
            story.append(Paragraph(value, styles["Heading2"]))
        elif kind == "paragraph":
            story.append(Paragraph(value, styles["BodyText"]))
        else:
            table = Table(value)
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
            story += [table, Spacer(1, 6)]
    SimpleDocTemplate(str(out), pagesize=LETTER, title=blocks[0][1] if blocks else "").build(story)


def main() -> None:
    for source in sorted(EXAMPLES.glob("*/source/*.md")):
        brain, sensitivity, filename = source.name.removesuffix(".md").split("__")
        out = source.parent.parent / "docs" / brain / sensitivity / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        blocks = parse(source.read_text())
        (write_pdf if out.suffix == ".pdf" else write_docx)(blocks, out)
        print(f"built {out.relative_to(EXAMPLES)}")


if __name__ == "__main__":
    main()
