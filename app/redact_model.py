import fitz  # PyMuPDF
import re
import io
from transformers import pipeline

ner = pipeline("token-classification", model="dslim/bert-base-NER", grouped_entities=True)

EMAIL_RE = re.compile(r"[a-zA-Z0-9.\-_+%]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

def redact_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """
    Takes raw PDF bytes, redacts PII (email, phone, name, org, location),
    and returns redacted PDF bytes — no saving on disk.
    """
    doc = fitz.open("pdf", pdf_bytes)

    for page in doc:
        words = page.get_text("words")
        for w in words:
            x0, y0, x1, y1, word, *_ = w
            replace = False

            if EMAIL_RE.fullmatch(word) or PHONE_RE.fullmatch(word):
                replace = True

            if not replace:
                try:
                    ents = ner(word)
                    for ent in ents:
                        if ent["entity_group"] in ["PER", "PERSON", "LOC"]:
                            replace = True
                            break
                except Exception:
                    pass

            if replace:
                rect = fitz.Rect(x0, y0, x1, y1)
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                page.insert_textbox(rect, "***", fontsize=10, color=(0, 0, 0), align=1)

    output_stream = io.BytesIO()
    doc.save(output_stream)
    doc.close()
    output_stream.seek(0)

    return output_stream.getvalue()
