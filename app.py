import streamlit as st
import sys

try:
    from logger import (
        log_query,
        log_answer,
        log_similarity,
        log_status
    )
except Exception as e:
    print(f"Error importing logger: {e}", file=sys.stderr)
    # Define dummy functions if logger fails
    def log_query(q):
        pass
    def log_answer(a):
        pass
    def log_similarity(s):
        pass
    def log_status(st_val):
        pass

# Configure the page immediately
try:
    st.set_page_config(
        page_title="Resume Search using RAG",
        page_icon="📄",
        layout="wide"
    )
except Exception as e:
    print(f"Error in set_page_config: {e}", file=sys.stderr)

# Display title and description immediately
st.title("📄 Resume Search using RAG")

st.write(
    "Ask questions about the resumes stored in the vector database."
)

st.divider()

# Now import heavy dependencies after UI is configured
try:
    from rag import ask_question
except Exception as e:
    st.error(f"Error loading RAG module: {e}")
    st.stop()

try:
    from evaluation import evaluate_answer
except Exception as e:
    st.error(f"Error loading Evaluation module: {e}")
    st.stop()

# UI interaction
try:
    query = st.text_input(
        "Enter your question"
    )

    if st.button("Search"):

        if not query.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching resumes..."):
                try:
                    # Generate Answer
                    answer = ask_question(query)

                    # Evaluate
                    result = evaluate_answer(
                        query,
                        answer
                    )

                    # Logging
                    log_query(query)
                    log_answer(answer)
                    log_similarity(result["similarity"])
                    log_status(result["status"])

                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    raise

            # Display Answer
            st.success("Generated Answer")
            st.write(answer)
            st.divider()

            # Ground Truth
            st.subheader("Ground Truth")
            st.write(result["ground_truth"])
            st.divider()

            # Candidate
            st.subheader("Candidate")
            st.write(result["candidate_name"])
            st.divider()

            # Similarity
            st.subheader("Similarity Score")
            st.metric(
                label="Similarity",
                value=f'{result["similarity"]}%'
            )
            st.divider()

            # Status
            st.subheader("Evaluation")

            if result["status"] == "Excellent":
                st.success(result["status"])

            elif result["status"] == "Good":
                st.info(result["status"])

            elif result["status"] == "Average":
                st.warning(result["status"])

            else:
                st.error(result["status"])

except Exception as e:
    st.error(f"Application error: {e}")
    import traceback
    st.text(traceback.format_exc())
