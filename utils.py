import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)


def get_file_extension(file_path):
    """
    Returns file extension.
    Example:
        resume.pdf -> .pdf
    """
    return Path(file_path).suffix.lower()


def ensure_directory(folder_path):
    """
    Creates folder if it doesn't exist.
    """
    os.makedirs(folder_path, exist_ok=True)


def print_separator():
    print("=" * 70)


def load_resume(file_path):
    """
    Load a single PDF or DOCX resume and return LangChain Documents.
    """

    extension = get_file_extension(file_path)

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError(f"Unsupported file format: {extension}")

    return loader.load()