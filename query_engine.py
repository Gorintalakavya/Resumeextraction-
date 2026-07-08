import os
import sys
from dotenv import load_dotenv

from config import VECTOR_DB_PATH, EMBEDDING_MODEL
from config import BASE_DIR
import traceback

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
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store from {VECTOR_DB_PATH}: {e}")

    try:
        # Create a LangChain retriever out of the vectorstore. Use similarity search.
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        # keep a reference to the underlying vectorstore for fallbacks
        try:
            retriever._vectorstore = vectorstore
        except Exception:
            pass
    except Exception as e:
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

        # Support different retriever method names depending on LangChain version
        if hasattr(retr, "get_relevant_documents"):
            docs = retr.get_relevant_documents(question)
        elif hasattr(retr, "retrieve"):
            docs = retr.retrieve(question)
        elif callable(retr):
            docs = retr(question)
        else:
            raise RuntimeError("Retriever does not support retrieval methods")

        # If no docs were returned, try falling back to underlying vectorstore methods
        if not docs and hasattr(retr, "_vectorstore"):
            try:
                vs = retr._vectorstore
                if hasattr(vs, "similarity_search"):
                    docs = vs.similarity_search(question, k=5)
                elif hasattr(vs, "similarity_search_with_score"):
                    docs_with_scores = vs.similarity_search_with_score(question, k=5)
                    # extract only documents
                    docs = [d for d, _ in docs_with_scores]
            except Exception:
                # ignore fallback errors, will return empty
                pass

    except Exception as exc:
        print("Error retrieving documents:\n" + traceback.format_exc(), file=sys.stderr)
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
        results.append(
            {
                "candidate_name": doc.metadata.get("candidate_name", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "content": doc.page_content,
            }
        )

    return results