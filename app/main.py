from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.redact_model import redact_pdf_bytes
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient
import os
import tempfile
import json
import re
from io import BytesIO
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Google Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME", "resume_parser")
collection_name = os.getenv("COLLECTION_NAME", "parsed_resumes")

mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]
collection = db[collection_name]

# FastAPI app
app = FastAPI(title="Resume Parser API (Gemini + MongoDB)")

class FilePath(BaseModel):
    pdf_path: str


@app.get("/")
def home():
    return {"message": "POST /extract → parses a resume PDF and saves structured data to MongoDB"}


@app.post("/extract")
async def extract_resume(input_data: FilePath):
    pdf_path = input_data.pdf_path

    # Validate file
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Step 1: Read PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Step 2: Redact sensitive info
        redacted_bytes = redact_pdf_bytes(pdf_bytes)

        # Step 3: Save temp file for Gemini upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(redacted_bytes)
            temp_path = temp_file.name

        # Step 4: Upload file to Gemini
        uploaded_pdf = client.files.upload(file=temp_path)

        # Step 5: Strong, structured prompt
        prompt = """
        You are an intelligent Resume Parser AI.
        Your goal: Read and analyze the uploaded resume and extract every meaningful section into a clean, structured JSON format.

        ⚙️ Guidelines:
        - Output **only JSON**, no markdown, no explanations.
        - Include all sections present in the resume, even if not explicitly listed below.
        - Keep keys and formatting consistent.
        - Group content logically (titles, subtitles, descriptions, bullet points).
        - Preserve as much information as possible.

        🧩 Example Output Structure:
        {
          "name": "",
          "contact": "",
          "summary": "",
          "education": [
            {"title": "", "sub_title": "", "description": ""}
          ],
          "skills": [],
          "experience": [
            {"title": "", "sub_title": "", "description": ""}
          ],
          "projects": [
            {"title": "", "description": ""}
          ],
          "certifications": [
            {"title": "", "description": ""}
          ],
          "additional_sections": {
            "awards": [],
            "publications": [],
            "languages": [],
            "volunteer_experience": []
          }
        }

        🧠 Notes:
        - Use empty strings or arrays if a section isn’t found.
        - Avoid including extra text outside the JSON.
        - The goal is to capture *all* resume details in clean, structured JSON.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, uploaded_pdf]
        )

        # Step 6: Clean Gemini’s response
        raw_text = response.text.strip()
        raw_text = re.sub(r"^```json|```$", "", raw_text).strip()

        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed_json = {"raw_output": raw_text, "note": "Unable to parse JSON perfectly"}

        # Step 7: Save to MongoDB
        record = {
            "filename": os.path.basename(pdf_path),
            "parsed_resume": parsed_json,
            "uploaded_at": datetime.utcnow()
        }
        inserted_id = collection.insert_one(record).inserted_id

        # Step 8: Convert to downloadable JSON file
        json_bytes = json.dumps(parsed_json, indent=4).encode("utf-8")
        json_stream = BytesIO(json_bytes)

        # Cleanup temp
        os.remove(temp_path)

        # Step 9: Return confirmation + download
        return StreamingResponse(
            json_stream,
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=parsed_resume.json",
                "X-Mongo-ID": str(inserted_id)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

