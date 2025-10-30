import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")

client = MongoClient(MONGO_URI)
db = client["resume_db"]

def ping_db():
    try:
        client.admin.command("ping")
        print("MongoDB connection successful.")
    except Exception as e:
        print("MongoDB connection failed:", e)
