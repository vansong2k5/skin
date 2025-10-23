from pathlib import Path
from dotenv import load_dotenv
import os

DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(DOTENV_PATH)  # nạp đúng file .env cạnh config.py

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env (backend/config/.env)")
