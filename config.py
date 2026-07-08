import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Gemini Configuration
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Names
# Use a Gemini embedding model known to support embedContent. The previous
# value ('models/embedding-001') caused a NOT_FOUND error from the provider.
# If your account supports a different embedding model, replace this value.
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESUME_FOLDER = os.path.join(BASE_DIR, "Resumes")

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

GROUND_TRUTH_FILE = os.path.join(
    BASE_DIR,
    "groundtruth",
    "Rag_Resumes (Responses).xlsx"
)

OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)