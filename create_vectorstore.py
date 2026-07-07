import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")


# =====================================================
# Folder Paths
# =====================================================

RESUME_FOLDER = "Resumes"
VECTOR_DB_PATH = "vector_store"


# =====================================================
# Load Documents
# =====================================================

def load_documents():

    documents = []

    resume_path = Path(RESUME_FOLDER)

    if not resume_path.exists():
        raise FileNotFoundError(f"{RESUME_FOLDER} folder not found.")

    for file in resume_path.iterdir():

        if file.suffix.lower() == ".pdf":

            print(f"Loading PDF : {file.name}")

            loader = PyPDFLoader(str(file))
            docs = loader.load()

        elif file.suffix.lower() == ".docx":

            print(f"Loading DOCX : {file.name}")

            loader = Docx2txtLoader(str(file))
            docs = loader.load()

        else:

            print(f"Skipping Unsupported File : {file.name}")
            continue

        for doc in docs:

            doc.metadata["candidate_name"] = file.stem
            doc.metadata["file_name"] = file.name
            doc.metadata["file_type"] = file.suffix.lower()

        documents.extend(docs)

    print(f"\nTotal Documents Loaded : {len(documents)}")

    return documents


# =====================================================
# Split Documents
# =====================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks Created : {len(chunks)}")

    return chunks


# =====================================================
# Create Vector Store
# =====================================================

def create_vectorstore(chunks):

    print("\nCreating Gemini Embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    vectorstore.save_local(VECTOR_DB_PATH)

    print("\nVector Store Created Successfully!")


# =====================================================
# Main Function
# =====================================================

def main():

    print("=" * 60)
    print("Resume Vector Store Creation Started")
    print("=" * 60)

    documents = load_documents()

    chunks = split_documents(documents)

    create_vectorstore(chunks)

    print("\nProcess Completed Successfully.")


if __name__ == "__main__":
    main()