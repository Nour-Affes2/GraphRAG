import fitz 
import os
from pathlib import Path
import json
from tqdm import tqdm


PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/texts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_pages = []
    for page in doc:
        text = page.get_text("text")
        text_pages.append(text)
    doc.close()
    full_text = "\n".join(text_pages)
    return full_text

def process_all():
    #used this to debug and see whether the program sees the pdfs or not
    #print("Looking for PDFs in:", PDF_DIR.resolve())
    #print(list(PDF_DIR.glob("*.pdf")))

    for pdf in tqdm(list(PDF_DIR.glob("*.pdf"))):
        try:
            text = extract_text_from_pdf(pdf)
            out_file = OUT_DIR / (pdf.stem + ".json")
            meta = {"pdf_filename": pdf.name, "source": "arXiv"}
            payload = {"metadata": meta, "text": text}
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        except Exception as e:
            print("Error with:", pdf, e)

if __name__ == "__main__":
    process_all()