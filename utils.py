
import os
from pathlib import Path


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