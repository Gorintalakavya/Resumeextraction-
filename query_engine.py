import os
import sys
from dotenv import load_dotenv

from config import VECTOR_DB_PATH, EMBEDDING_MODEL
from config import BASE_DIR
import traceback
from logger import logger

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

retriever = None


def _get_retriever():
    """
    Create the retriever lazily so Streamlit can render immediately.
    """
    global retriever

    if retriever is not None:
        return retriever

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    # Log configured vector DB path for diagnostics
    logger.info("Configured VECTOR_DB_PATH: %s", VECTOR_DB_PATH)

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        raise ImportError(f"Failed to import FAISS or GoogleGenerativeAIEmbeddings: {e}")

    try:
        # Initialize embeddings using the configured embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        err = str(e)
        logger.exception("Failed to initialize embeddings: %s", err)
        if 'not found' in err.lower() or 'not_found' in err.lower():
            raise RuntimeError(
                f"Embedding model {EMBEDDING_MODEL!r} not found or unsupported: {err}. "
                "Verify `EMBEDDING_MODEL` in config.py and your GOOGLE_API_KEY, or list provider models to find a compatible embedding model."
            )
        raise RuntimeError(f"Failed to initialize embeddings: {e}")

    try:
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector database not found at {VECTOR_DB_PATH}")

        # Load the persisted FAISS vectorstore
        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Loaded FAISS vectorstore from %s", VECTOR_DB_PATH)
        logger.info("FAISS vectorstore load successful: True")
        # Quick runtime validation: ensure embeddings client and vectorstore are compatible.
        try:
            test_text = "test"
            emb_vector = None
            if hasattr(embeddings, 'embed_documents'):
                emb_result = embeddings.embed_documents([test_text])
                emb_vector = emb_result[0] if emb_result else None
            elif hasattr(embeddings, 'embed_query'):
                emb_vector = embeddings.embed_query(test_text)
            elif hasattr(embeddings, 'embed'):
                emb_result = embeddings.embed([test_text])
                emb_vector = emb_result[0] if emb_result else None

            if emb_vector is not None:
                # Try a vector-based similarity search if available
                try:
                    if hasattr(vectorstore, 'similarity_search_by_vector'):
                        _ = vectorstore.similarity_search_by_vector(emb_vector, k=1)
                    elif hasattr(vectorstore, 'similarity_search_with_score'):
                        _ = vectorstore.similarity_search_with_score(test_text, k=1)
                    else:
                        # fallback to text-based search which will call embeddings again
                        _ = vectorstore.similarity_search(test_text, k=1)
                except Exception as e:
                    logger.exception("Vectorstore retrieval test failed: %s", e)
                    raise RuntimeError(
                        "Vectorstore appears incompatible with the current embedding model. "
                        "Recreate the vectorstore by running: python create_vectorstore.py"
                    )
            else:
                logger.warning("Could not create a test embedding vector; skipping vectorstore retrieval test")
        except Exception as e:
            logger.exception("Runtime validation of embeddings/vectorstore failed: %s", e)
            raise
        # Determine if vectorstore contains vectors
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
            # fallback: attempt to estimate via metadata map length
            try:
                if hasattr(vectorstore, 'index_to_docstore_id'):
                    vector_count = len(vectorstore.index_to_docstore_id)
            except Exception:
                vector_count = None

        logger.info("Detected %s vectors in loaded vectorstore", vector_count if vector_count is not None else "unknown")

        if vector_count == 0:
            logger.error("Loaded vectorstore contains 0 vectors")
            raise RuntimeError("Vectorstore is empty. Recreate it using create_vectorstore.py")
    except Exception as e:
        logger.exception("Failed to load FAISS vectorstore from %s", VECTOR_DB_PATH)
        raise RuntimeError(f"Failed to load vector store from {VECTOR_DB_PATH}: {e}")

    try:
        # Create a LangChain retriever out of the vectorstore. Use similarity search.
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        logger.info("Retriever created with search_type=similarity and k=5")
        try:
            logger.info("Retriever class: %s", type(retriever).__name__)
        except Exception:
            logger.info("Retriever class: unknown")
        # keep a reference to the underlying vectorstore for fallbacks
        try:
            retriever._vectorstore = vectorstore
        except Exception:
            logger.debug("Could not attach underlying vectorstore to retriever")
    except Exception as e:
        logger.exception("Failed to create retriever")
        raise RuntimeError(f"Failed to create retriever: {e}")

    return retriever


# ===========================================
# Retrieve Documents
# ===========================================

def retrieve_documents(question):
    """
    Retrieve top matching resume chunks.
    """
    try:
        retr = _get_retriever()

        logger.info("Retrieving documents for question: %s", question)

        # Support latest LangChain API by preferring invoke(), then fallback to older methods.
        if hasattr(retr, "invoke"):
            docs = retr.invoke(question)
        elif hasattr(retr, "get_relevant_documents"):
            docs = retr.get_relevant_documents(question)
        elif hasattr(retr, "retrieve"):
            docs = retr.retrieve(question)
        elif callable(retr):
            docs = retr(question)
        else:
            raise RuntimeError("Retriever does not support retrieval methods")

        retrieved_count = len(docs) if docs else 0
        logger.info("Retriever returned %d documents", retrieved_count)

        # Log metadata and content snippet for each retrieved document
        if docs:
            try:
                for i, d in enumerate(docs, start=1):
                    meta = getattr(d, 'metadata', {}) or {}
                    # metadata full dump
                    logger.info("Doc %d metadata: %s", i, meta)
                    # first 200 characters of page_content
                    content = getattr(d, 'page_content', '') or ''
                    snippet = content[:200]
                    logger.info("Doc %d content snippet: %s", i, snippet)
            except Exception:
                logger.exception("Failed to log individual document metadata or content")
        else:
            # Explicitly log zero documents retrieved
            logger.info("Retriever returned zero documents for question: %s", question)

        # If no docs were returned, try falling back to underlying vectorstore methods
        if (not docs or len(docs) == 0) and hasattr(retr, "_vectorstore"):
            try:
                logger.info("Attempting fallback similarity_search on underlying vectorstore")
                vs = retr._vectorstore
                if hasattr(vs, "similarity_search"):
                    docs = vs.similarity_search(question, k=5)
                elif hasattr(vs, "similarity_search_with_score"):
                    docs_with_scores = vs.similarity_search_with_score(question, k=5)
                    # extract only documents
                    docs = [d for d, _ in docs_with_scores]
                logger.info("Fallback returned %d documents", len(docs) if docs else 0)
            except Exception:
                logger.exception("Fallback similarity_search failed")
                # ignore fallback errors, will return empty
                pass

    except Exception as exc:
        logger.exception("Error retrieving documents: %s", exc)
        return []

    return docs


def retrieve_with_metadata(question):
    """
    Retrieve chunks along with metadata.
    """
    try:
        retr = _get_retriever()

        if hasattr(retr, "get_relevant_documents"):
            docs = retr.get_relevant_documents(question)
        elif hasattr(retr, "retrieve"):
            docs = retr.retrieve(question)
        elif callable(retr):
            docs = retr(question)
        else:
            raise RuntimeError("Retriever does not support retrieval methods")
    except Exception:
        return []

    results = []

    for doc in docs:
        try:
            # Log each doc's metadata for debugging
            meta = getattr(doc, 'metadata', {}) or {}
            logger.debug("retrieve_with_metadata - metadata: %s", meta)
        except Exception:
            logger.debug("Failed to read document metadata in retrieve_with_metadata", exc_info=True)
        results.append(
            {
                "candidate_name": doc.metadata.get("candidate_name", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "content": doc.page_content,
            }
        )

    return results