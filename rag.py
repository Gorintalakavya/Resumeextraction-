import os
import sys
from dotenv import load_dotenv

from query_engine import retrieve_documents
from config import LLM_MODEL
from prompts import SYSTEM_PROMPT
from logger import logger

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = None


def _get_llm():
    """Create the Gemini client lazily so the app stays responsive."""
    global llm

    if llm is not None:
        return llm

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment when initializing LLM")
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(f"Failed to import ChatGoogleGenerativeAI: {e}")

    try:
        logger.info("Initializing Gemini LLM with model: %s", LLM_MODEL)
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Gemini LLM initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize Gemini LLM")
        raise RuntimeError(f"Failed to initialize Gemini LLM: {e}")

    return llm


# =====================================================
# Ask Question Function
# =====================================================

def ask_question(query):
    """
    Accepts a user question, retrieves relevant resume chunks,
    and generates the final answer using Gemini.
    """

    try:
        logger.info("User question: %s", query)

        docs = retrieve_documents(query)

        logger.info("Documents retrieved for query '%s': %d", query, len(docs) if docs else 0)  # Log retrieval result

        if not docs:
            logger.info("No documents returned by retriever for query: %s", query)
            return "No relevant information found in the resumes."

        context = "\n\n".join(doc.page_content for doc in docs)

        logger.info("Retrieved context length (chars): %d", len(context))
        logger.debug("Context preview: %s", (context[:1000] + '...') if len(context) > 1000 else context)

        prompt = f"""
{SYSTEM_PROMPT}

Resume Context:
{context}

User Question:
{query}

Instructions:
- Answer ONLY using the resume context.
- Mention candidate names whenever possible.
- Mention relevant skills, education, projects, certifications and experience.
- If the answer is not available in the resumes, reply:
  "The information is not available in the resumes."
- Do not hallucinate or create information.

Answer:
"""

        logger.info("Invoking LLM to generate answer; prompt length (chars): %d", len(prompt))
        response = _get_llm().invoke(prompt)
        try:
            content = response.content
            logger.info("LLM response length (chars): %d", len(content) if content else 0)
        except Exception:
            content = str(response)
            logger.debug("LLM response could not extract .content, using str(response)")

        logger.info("Generated response: %s", (content[:1000] + '...') if content and len(content) > 1000 else content)

        return content

    except Exception as exc:
        logger.exception("Error generating answer for query: %s", query)
        return f"Unable to generate an answer. Error: {exc}"


# =====================================================
# Terminal Testing
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Resume Search using RAG")
    print("=" * 60)

    while True:

        query = input("\nAsk your resume question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("\nExiting Resume RAG...")
            break

        try:
            answer = ask_question(query)

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)
            print(answer)

        except Exception as e:
            print(f"\nError: {e}")