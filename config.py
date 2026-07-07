

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Gemini Configuration
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Names
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-flash"

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESUME_FOLDER = os.path.join(BASE_DIR, "Resumes")

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectorstore")

GROUND_TRUTH_FILE = os.path.join(
    BASE_DIR,
    "groundtruth",
    "Rag_Resumes (Responses).xlsx"
)

OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

# Create output folder automatically
os.makedirs(OUTPUT_FOLDER, exist_ok=True)