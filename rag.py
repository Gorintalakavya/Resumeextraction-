import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# -------------------------------
# Load API Key
# -------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Load Embedding Model
# -------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# -------------------------------
# Load Vector Store
# -------------------------------
vectorstore = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# -------------------------------
# Create Retriever
# -------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# -------------------------------
# Load Gemini LLM
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


# =========================================================
# Function used by both Streamlit and terminal
# =========================================================
def ask_question(query):

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an AI Resume Screening Assistant.

Use only the resume information provided below.

Resume Context:
{context}

User Question:
{query}

Instructions:
- Identify suitable candidates.
- Mention candidate names.
- Mention matching skills and experience.
- If no candidate matches, say no matching candidate found.
- Do not make up information.

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# Terminal Mode
# =========================================================
if __name__ == "__main__":

    while True:

        query = input("\nAsk your resume question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = ask_question(query)

        print("\n==============================")
        print("FINAL ANSWER")
        print("==============================")
        print(answer)