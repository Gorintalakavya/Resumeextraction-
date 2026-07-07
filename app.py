import streamlit as st
from rag import ask_question

st.set_page_config(
    page_title="Resume Search using RAG",
    page_icon=":page_facing_up:",
    layout="wide"
)

st.title(":page_facing_up: Resume Search using RAG")

st.write("Ask questions about the resumes stored in the vector database.")

query = st.text_input("Enter your question")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            answer = ask_question(query)

        st.success("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")