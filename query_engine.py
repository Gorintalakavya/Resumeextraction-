import os
from dotenv import load_dotenv

from config import VECTOR_DB_PATH, EMBEDDING_MODEL

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

    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return retriever


# ===========================================
# Retrieve Documents
# ===========================================

def retrieve_documents(question):
    """
    Retrieve top matching resume chunks.
    """
    try:
        docs = _get_retriever().invoke(question)
    except Exception:
        return []

    return docs


def retrieve_with_metadata(question):
    """
    Retrieve chunks along with metadata.
    """
    try:
        docs = _get_retriever().invoke(question)
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