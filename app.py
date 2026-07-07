import streamlit as st

from rag import ask_question
from evaluation import evaluate_answer

from logger import (
    log_query,
    log_answer,
    log_similarity,
    log_status
)

st.set_page_config(
    page_title="Resume Search using RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Search using RAG")

st.write(
    "Ask questions about the resumes stored in the vector database."
)

st.divider()

query = st.text_input(
    "Enter your question"
)

if st.button("Search"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching resumes..."):

            # -------------------------------
            # Generate Answer
            # -------------------------------

            answer = ask_question(query)

            # -------------------------------
            # Evaluate
            # -------------------------------

            result = evaluate_answer(
                query,
                answer
            )

            # -------------------------------
            # Logging
            # -------------------------------

            log_query(query)
            log_answer(answer)
            log_similarity(result["similarity"])
            log_status(result["status"])

        # -------------------------------
        # Display Answer
        # -------------------------------

        st.success("Generated Answer")

        st.write(answer)

        st.divider()

        # -------------------------------
        # Ground Truth
        # -------------------------------

        st.subheader("Ground Truth")

        st.write(result["ground_truth"])

        st.divider()

        # -------------------------------
        # Candidate
        # -------------------------------

        st.subheader("Candidate")

        st.write(result["candidate_name"])

        st.divider()

        # -------------------------------
        # Similarity
        # -------------------------------

        st.subheader("Similarity Score")

        st.metric(
            label="Similarity",
            value=f'{result["similarity"]}%'
        )

        st.divider()

        # -------------------------------
        # Status
        # -------------------------------

        st.subheader("Evaluation")

        if result["status"] == "Excellent":

            st.success(result["status"])

        elif result["status"] == "Good":

            st.info(result["status"])

        elif result["status"] == "Average":

            st.warning(result["status"])

        else:

            st.error(result["status"])