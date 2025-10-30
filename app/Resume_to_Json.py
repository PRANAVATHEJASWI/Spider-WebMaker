from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.redact_model import redact_pdf_bytes
from dotenv import load_dotenv
from google import genai
import os

# Load environment variables
load_dotenv()

# Initialize Google Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Resume Redactor & Parser API")

class FilePath(BaseModel):
    pdf_path: str


@app.get("/")
def home():
    return {"message": "Welcome to the Resume Parser API — POST /extract with a PDF path"}


@app.post("/extract")
async def extract_resume(input_data: FilePath):
    pdf_path = input_data.pdf_path

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Step 1: Read file in bytes
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Step 2: Redact sensitive info
        redacted_bytes = redact_pdf_bytes(pdf_bytes)

        # Step 3: Send to Gemini for structured extraction
        prompt = """
        You are a resume parser AI.
        Convert the given resume into clean JSON with the following structure:
        {
          "education": [],
          "skills": [],
          "experience": [],
          "projects": [],
          "certifications": [],
          "summary": ""
        }
        Please note if there are any other sections include that tooo
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                {
                    "mime_type": "application/pdf",
                    "data": redacted_bytes
                }
            ]
        )

        return JSONResponse(content={"parsed_resume": response.text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
