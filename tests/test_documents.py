import io

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from secondbrain.documents import chunk, chunk_document, extract_text
from tests.conftest import ADMIN_TOKEN


def make_docx() -> bytes:
    doc = Document()
    doc.add_heading("Employee Handbook", 0)
    doc.add_heading("Annual leave", 1)
    doc.add_paragraph("Full-time employees get 22 days of paid leave per year.")
    doc.add_heading("Expenses", 1)
    doc.add_paragraph("Meals on business travel are covered up to 40 dollars per day.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Level", "Hotel cap"
    table.cell(1, 0).text, table.cell(1, 1).text = "Manager", "180 dollars"
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def make_pdf() -> bytes:
    styles = getSampleStyleSheet()
    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4).build([
        Paragraph("Returns Policy", styles["Title"]),
        Paragraph("Return window", styles["Heading2"]),
        Paragraph("Customers can return unused items within 30 days of delivery for a full refund.", styles["BodyText"]),
        Paragraph("Damaged items", styles["Heading2"]),
        Paragraph("Items that arrive damaged are replaced free of charge within 48 hours.", styles["BodyText"]),
    ])
    return out.getvalue()


def test_docx_keeps_headings_and_tables():
    text = extract_text("handbook.docx", make_docx())
    assert "# Employee Handbook" in text and "## Annual leave" in text
    assert "Manager | 180 dollars" in text
    chunks = chunk_document("handbook.docx", text)
    assert chunks[0].startswith("Employee Handbook › Annual leave")
    assert any(c.startswith("Employee Handbook › Expenses") and "180 dollars" in c for c in chunks)


def test_pdf_text_is_rebuilt_into_sections():
    text = extract_text("returns.pdf", make_pdf())
    assert "30 days of delivery" in text
    labels = [c.split("\n")[0] for c in chunk_document("returns.pdf", text)]
    assert any("Return window" in label for label in labels)
    assert any("Damaged items" in label for label in labels)


def test_long_paragraph_is_split_between_sentences():
    text = " ".join(["This is one sentence about policy."] * 80)
    pieces = chunk(text, max_chars=500)
    assert len(pieces) > 1 and all(len(p) <= 500 for p in pieces)


def test_upload_word_and_pdf_through_admin_api(client):
    headers = {"X-Admin-Token": ADMIN_TOKEN}
    response = client.post(
        "/admin/brains/root.acme.hr/documents", headers=headers,
        files=[("files", ("handbook.docx", make_docx(), "application/octet-stream")),
               ("files", ("returns.pdf", make_pdf(), "application/pdf"))],
    )
    assert response.status_code == 200, response.text
    assert response.json()["chunks"] >= 4
    bad = client.post("/admin/brains/root.acme.hr/documents", headers=headers,
                      files=[("files", ("sheet.xlsx", b"x", "application/octet-stream"))])
    assert bad.status_code == 400
    graph = client.get("/admin/graph", headers=headers).json()
    contents = [n["props"].get("content", "") for n in graph["nodes"] if n["props"].get("brain") == "root.acme.hr"]
    assert any(c.startswith("Employee Handbook › Expenses") and "180 dollars" in c for c in contents)
    assert any("Return window" in c and "30 days of delivery" in c for c in contents)
