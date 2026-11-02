"""
Script to generate publication-grade PDF of ECG_GUARDIAN_COMPLETE_SYSTEM_VALIDATION_REPORT.md
using ReportLab Platypus.
"""

import os
import re
import html
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Preformatted,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically for header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 762, "ECG Guardian — System Technical & Clinical Validation Dossier")
            self.drawRightString(576, 762, "Confidential & Audited")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_str)
        self.drawString(36, 25, "CDSCO Medical Device Rules 2017 Class B | IEC 62304 Class B Technical Dossier")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 35, 576, 35)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Safely escapes HTML and formats markdown inline elements."""
    # First, handle backtick code blocks by placeholder
    placeholders = []
    def repl_code(m):
        code_txt = html.escape(m.group(1))
        placeholders.append(f"<font name='Courier' size='7'>{code_txt}</font>")
        return f"___CODE_{len(placeholders)-1}___"

    text = re.sub(r"`(.*?)`", repl_code, text)

    # Escape raw HTML characters
    text = html.escape(text)

    # Format bold, italic, links
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"<u>\1</u>", text)

    # Restore code blocks
    for idx, ph in enumerate(placeholders):
        text = text.replace(f"___CODE_{idx}___", ph)

    return text


def markdown_to_pdf(md_path: Path, pdf_path: Path):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e1b4b"),
        spaceAfter=8,
    )

    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#4338ca"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor("#1e293b"),
    )

    story = []
    lines = text.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip()

        # Empty line
        if not line:
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^---+$", line):
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=2, spaceAfter=3))
            i += 1
            continue

        # Code block
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n and lines[i].startswith("```"):
                i += 1
            code_text = "\n".join(code_lines)
            story.append(Preformatted(code_text, code_style))
            story.append(Spacer(1, 3))
            continue

        # Table block
        if line.startswith("|") and line.endswith("|"):
            table_lines = []
            while i < n and lines[i].startswith("|") and lines[i].endswith("|"):
                table_lines.append(lines[i])
                i += 1

            # Parse markdown table
            parsed_rows = []
            for r_idx, tline in enumerate(table_lines):
                # Check if it's separator row | :--- | :--- |
                if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", tline):
                    continue
                cells = [c.strip() for c in tline.split("|")[1:-1]]
                parsed_rows.append(cells)

            if parsed_rows:
                # Wrap each cell in a Paragraph for auto text wrapping
                num_cols = max(len(r) for r in parsed_rows)
                formatted_table_data = []

                for row_idx, row in enumerate(parsed_rows):
                    # Pad cells if needed
                    while len(row) < num_cols:
                        row.append("")
                    formatted_row = []
                    for c in row:
                        c_clean = clean_markdown_inline(c)
                        st = table_header_style if row_idx == 0 else table_cell_style
                        formatted_row.append(Paragraph(c_clean, st))
                    formatted_table_data.append(formatted_row)

                # Available width: 540 pt
                col_width = 540.0 / num_cols
                t = Table(formatted_table_data, colWidths=[col_width] * num_cols, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338ca")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
                ]))
                story.append(t)
                story.append(Spacer(1, 5))
            continue

        # Headings
        if line.startswith("# "):
            title_text = clean_markdown_inline(line[2:].strip())
            story.append(Paragraph(title_text, title_style))
            i += 1
            continue
        elif line.startswith("## "):
            h1_text = clean_markdown_inline(line[3:].strip())
            story.append(Paragraph(h1_text, h1_style))
            i += 1
            continue
        elif line.startswith("### "):
            h2_text = clean_markdown_inline(line[4:].strip())
            story.append(Paragraph(h2_text, h2_style))
            i += 1
            continue
        elif line.startswith("#### "):
            h3_text = clean_markdown_inline(line[5:].strip())
            story.append(Paragraph(h3_text, h3_style))
            i += 1
            continue

        # Bullet list or normal text
        clean_line = line
        is_bullet = False
        if clean_line.startswith("- ") or clean_line.startswith("* "):
            clean_line = "&bull; " + clean_markdown_inline(clean_line[2:])
            is_bullet = True
        elif re.match(r"^\d+\.\s+", clean_line):
            m = re.match(r"^(\d+\.)\s+(.*)", clean_line)
            clean_line = f"<b>{m.group(1)}</b> " + clean_markdown_inline(m.group(2))
        else:
            clean_line = clean_markdown_inline(clean_line)

        story.append(Paragraph(clean_line, body_style))
        i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Validation Report PDF generated successfully: {pdf_path}")


if __name__ == "__main__":
    proj_dir = Path(__file__).resolve().parent.parent
    md_file = proj_dir / "ECG_GUARDIAN_COMPLETE_SYSTEM_VALIDATION_REPORT.md"
    pdf_file = proj_dir / "ECG_GUARDIAN_COMPLETE_SYSTEM_VALIDATION_REPORT.pdf"
    markdown_to_pdf(md_file, pdf_file)
