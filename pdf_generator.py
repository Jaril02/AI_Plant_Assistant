# pdf_generator.py
import os
from fpdf import FPDF

class PDFReport:
    def __init__(self, title="Plant Disease Report"):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.title = title

    def _safe(self, text):
        """Convert to str and replace unsupported chars"""
        if text is None:
            return ""
        s = str(text)
        replacements = {
            "•": "-",
            "🌱": "[plant]",
            "📄": "[report]",
            "🔍": "[search]",
            "💬": "[chat]",
            "⚠️": "[warn]",
            "✅": "[ok]",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s.encode("latin-1", "replace").decode("latin-1")

    def add_page(self):
        self.pdf.add_page()
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.cell(0, 10, self._safe(self.title), ln=True, align="C")
        self.pdf.ln(10)

    def add_image_and_text(self, image_path, disease_data, ai_summary="", ai_detailed="", ai_prevention=""):
        # Add image
        if image_path and os.path.exists(image_path):
            try:
                self.pdf.image(image_path, x=55, w=100)
                self.pdf.ln(85)
            except Exception as e:
                self.pdf.set_font("Helvetica", "", 12)
                self.pdf.multi_cell(0, 8, self._safe(f"[Image skipped: {e}]"))
                self.pdf.ln(5)

        # Disease Information
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.cell(0, 10, self._safe("Disease Information"), ln=True)
        self.pdf.set_font("Helvetica", "", 12)
        for key, value in disease_data.items():
            self.pdf.multi_cell(0, 8, self._safe(f"{key}: {value}"))
            self.pdf.ln(1)

        def add_section(title, body):
            if not body:
                return
            self.pdf.set_font("Helvetica", "B", 14)
            self.pdf.cell(0, 10, self._safe(title), ln=True)
            self.pdf.set_font("Helvetica", "", 12)
            self.pdf.multi_cell(0, 8, self._safe(body))
            self.pdf.ln(3)

        add_section("AI Summary", ai_summary)
        add_section("Detailed Analysis", ai_detailed)
        add_section("Prevention Guide", ai_prevention)

    def export_pdf(self, filename="plant_disease_report.pdf"):
        self.pdf.output(filename)
        return filename
