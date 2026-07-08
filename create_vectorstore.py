import os
from pathlib import Path

from dotenv import load_dotenv
from logger import logger

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL, VECTOR_DB_PATH, RESUME_FOLDER

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

RESUME_FOLDER = RESUME_FOLDER
# Use project-configured VECTOR_DB_PATH so vectorstore is saved where the app expects it
VECTOR_DB_PATH = VECTOR_DB_PATH


# =====================================================
# Load Documents
# =====================================================

def load_documents():

    documents = []
    logger.info("Starting to load resumes from folder: %s", RESUME_FOLDER)  # Log start of loading resumes

    resume_path = Path(RESUME_FOLDER)

    if not resume_path.exists():
        raise FileNotFoundError(f"{RESUME_FOLDER} folder not found.")

    processed_files = 0
    failed_files = []

    for file in sorted(resume_path.iterdir()):

        if file.suffix.lower() == ".pdf":

            logger.info("Loading PDF: %s", file.name)  # Log each file being loaded

            loader = PyPDFLoader(str(file))
            docs = loader.load()

        elif file.suffix.lower() == ".docx":

            logger.info("Loading DOCX: %s", file.name)  # Log each file being loaded

            loader = Docx2txtLoader(str(file))
            docs = loader.load()

        else:

            logger.info("Skipping unsupported file: %s", file.name)  # Log skipped files
            continue

        processed_files += 1

        docs_count = len(docs) if docs is not None else 0
        logger.info("File %s -> pages/documents loaded: %d", file.name, docs_count)

        if docs_count == 0:
            logger.warning("No pages/documents were extracted from file: %s", file.name)
            failed_files.append(file.name)

        for doc in docs:

            doc.metadata["candidate_name"] = file.stem
            doc.metadata["file_name"] = file.name
            doc.metadata["file_type"] = file.suffix.lower()

        documents.extend(docs)

    logger.info("Processed files: %d, failed files: %d", processed_files, len(failed_files))
    if failed_files:
        logger.info("Failed files: %s", failed_files)

    logger.info("Total documents loaded: %d", len(documents))  # Log total docs loaded

    # Raise if no resumes found
    if len(documents) == 0:
        logger.error("No resumes were loaded from folder: %s", RESUME_FOLDER)
        raise RuntimeError(f"No resumes found in folder: {RESUME_FOLDER}")

    # Also write a debug sample of first document metadata if available
    if documents:
        try:
            logger.debug("First document metadata: %s", documents[0].metadata)
        except Exception:
            pass

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

    logger.info("Total chunks created: %d", len(chunks))  # Log chunking result

    # Log an example chunk length for debugging
    if chunks:
        try:
            logger.debug("Example chunk size: %d", len(chunks[0].page_content))
        except Exception:
            pass

    # Raise if no chunks created
    if len(chunks) == 0:
        logger.error("No chunks were created from the loaded documents")
        raise RuntimeError("No chunks created from resumes. Check splitter configuration and document content.")

    return chunks


# =====================================================
# Create Vector Store
# =====================================================

def create_vectorstore(chunks):

    logger.info("Starting vectorstore creation using embedding model: %s", EMBEDDING_MODEL)  # Log start of vectorstore creation

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Embeddings client initialized successfully")  # Log embeddings init success
        # Attempt a small test embedding on first chunk content if available
        try:
            if chunks and len(chunks) > 0:
                embed_fn = None
                if hasattr(embeddings, 'embed_documents'):
                    embed_fn = embeddings.embed_documents
                elif hasattr(embeddings, 'embed_query'):
                    embed_fn = lambda docs: [embeddings.embed_query(d) for d in docs]
                elif hasattr(embeddings, 'embed'):
                    embed_fn = embeddings.embed

                if embed_fn:
                    try:
                        sample_text = chunks[0].page_content if hasattr(chunks[0], 'page_content') else str(chunks[0])
                        test_emb = embed_fn([sample_text])
                        logger.info("Test embedding created, length: %s", len(test_emb[0]) if test_emb and isinstance(test_emb[0], (list, tuple)) else 'unknown')
                    except Exception:
                        # If the embeddings API reports the model is not found or unsupported,
                        # raise a clear error so the user can correct `config.EMBEDDING_MODEL`.
                        import traceback as _tb
                        err = _tb.format_exc()
                        logger.exception("Test embedding call failed: %s", err)
                        if 'not found' in err.lower() or 'not_found' in err.lower() or 'notfound' in err.lower():
                            raise RuntimeError(
                                f"Embedding model {EMBEDDING_MODEL!r} not found or not supported for embeddings API. "
                                "Please verify `EMBEDDING_MODEL` in config.py and your GOOGLE_API_KEY, or call the provider's ModelService.ListModels to discover supported models."
                            )
                        # otherwise continue but log the failure
                        raise RuntimeError(f"Test embedding failed: {err}")
        except Exception:
            # non-fatal: continue
            pass
    except Exception as e:
        logger.exception("Failed to initialize embeddings for vectorstore creation")
        raise RuntimeError(f"Failed to initialize embeddings for vectorstore creation: {e}")

    try:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
        logger.info("FAISS vectorstore created from documents")  # Log vectorstore creation
    except Exception as e:
        # Provide clearer guidance if embedding model is invalid
        err_msg = str(e)
        logger.exception("Failed to create FAISS vectorstore from documents: %s", err_msg)
        if 'not found' in err_msg.lower() or 'not_found' in err_msg.lower():
            raise RuntimeError(
                f"Embedding model {EMBEDDING_MODEL!r} appears invalid or unsupported: {err_msg}. "
                "Update `EMBEDDING_MODEL` in config.py to a supported embedding model and ensure your GOOGLE_API_KEY is valid."
            )
        raise

    try:
        vectorstore.save_local(VECTOR_DB_PATH)
        logger.info("Vectorstore saved locally at %s", VECTOR_DB_PATH)  # Log where vectorstore is saved
    except Exception as e:
        logger.exception("Failed to save vectorstore to disk")
        raise

    # Validate persistent files exist (FAISS index files)
    try:
        index_faiss = os.path.join(VECTOR_DB_PATH, "index.faiss")
        index_pkl = os.path.join(VECTOR_DB_PATH, "index.pkl")
        files_found = []
        if os.path.exists(index_faiss):
            files_found.append(index_faiss)
        if os.path.exists(index_pkl):
            files_found.append(index_pkl)

        logger.info("Vectorstore files present: %s", files_found)

        if not files_found:
            # Some FAISS variants may use different filenames; list directory for debugging
            try:
                saved_files = os.listdir(VECTOR_DB_PATH)
            except Exception:
                saved_files = []
            logger.error("Expected FAISS files not found in %s. Found: %s", VECTOR_DB_PATH, saved_files)
            raise RuntimeError(f"Vectorstore files not found in {VECTOR_DB_PATH}")
    except Exception:
        raise

    # Try to determine number of vectors stored in the FAISS index
    vector_count = None
    try:
        if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
            vector_count = int(vectorstore.index.ntotal)
        elif hasattr(vectorstore, '_faiss_index') and hasattr(vectorstore._faiss_index, 'ntotal'):
            vector_count = int(vectorstore._faiss_index.ntotal)
        elif hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
            vector_count = len(vectorstore.docstore._dict)
    except Exception:
        vector_count = None

    if vector_count is None:
        # fallback to chunk count
        vector_count = len(chunks)

    logger.info("Number of vectors stored in vectorstore: %d", vector_count)

    if vector_count == 0:
        logger.error("Vectorstore appears to contain 0 vectors after creation")
        raise RuntimeError("Vectorstore is empty after creation")


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