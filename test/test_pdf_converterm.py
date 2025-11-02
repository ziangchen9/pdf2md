import os
import sys

# Ensure project root is on sys.path when running this test directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pdf_converter import PDFConverter


def test_pdf_converter():
    converter = PDFConverter.create_converter("config/config.ini")
    converter.convert_to_markdown(
        "data/pdfs/ClassicComputerScienceProblemsInPython.pdf"
    )


if __name__ == "__main__":
    test_pdf_converter()
