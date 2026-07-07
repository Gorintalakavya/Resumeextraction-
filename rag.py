import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)


# Load API key
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -------------------------------
# 1. Load Embedding Model
# -------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# -------------------------------
# 2. Load Existing Vector Store
# -------------------------------

vectorstore = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)


# -------------------------------
# 3. Create Retriever
# -------------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# -------------------------------
# 4. User Query
# -------------------------------

query = input("\nAsk your resume question: ")


# -------------------------------
# 5. Retrieval Step
# -------------------------------

docs = retriever.invoke(query)


print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================")

for i, doc in enumerate(docs):
    print(f"\nRESULT {i+1}")
    print("------------------------------")
    print(doc.page_content[:500])


# -------------------------------
# 6. Prepare Context
# -------------------------------

context = "\n\n".join(
    [
        doc.page_content
        for doc in docs
    ]
)


# -------------------------------
# 7. Load Gemini LLM
# -------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


# -------------------------------
# 8. Generation Prompt
# -------------------------------

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


# -------------------------------
# 9. Generate Final Answer
# -------------------------------

response = llm.invoke(prompt)


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(response.content)