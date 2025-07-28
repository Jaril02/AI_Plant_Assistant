# pdf_generator.py
from fpdf import FPDF
from PIL import Image

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Plant Disease Report", ln=True, align='C')
        self.ln(10)

    def add_image_and_text(self, image_path, disease_data, ai_summary=None, ai_detailed=None, ai_prevention=None):
        self.set_font("Arial", "", 12)

        # Add image
        if image_path:
            self.image(image_path, x=10, y=self.get_y(), w=60)
            self.ln(65)

        # Add disease details
        for key, value in disease_data.items():
            self.set_font("Arial", "B", 12)
            self.cell(40, 10, f"{key}:", ln=False)
            self.set_font("Arial", "", 12)
            self.multi_cell(0, 10, value)
            self.ln(1)

        # Add AI sections
        if ai_summary:
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "AI-Generated Summary", ln=True)
            self.set_font("Arial", "", 12)
            self.multi_cell(0, 10, ai_summary)
            self.ln()

        if ai_detailed:
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "AI-Generated Detailed Info", ln=True)
            self.set_font("Arial", "", 12)
            self.multi_cell(0, 10, ai_detailed)
            self.ln()

        if ai_prevention:
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "AI-Generated Prevention Techniques", ln=True)
            self.set_font("Arial", "", 12)
            self.multi_cell(0, 10, ai_prevention)
            self.ln()

    def export_pdf(self, output_path="plant_disease_report.pdf"):
        self.output(output_path)
        return output_path
