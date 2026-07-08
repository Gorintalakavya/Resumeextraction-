import os
import sys
import pandas as pd

from config import GROUND_TRUTH_FILE
from logger import logger


EXCEL_PATH = GROUND_TRUTH_FILE


def load_ground_truth():
    """
    Load the Ground Truth Excel file.
    """

    if not os.path.exists(EXCEL_PATH):
        logger.error("Ground Truth file not found: %s", EXCEL_PATH)
        raise FileNotFoundError(
            f"Ground Truth file not found: {EXCEL_PATH}"
        )

    try:
        logger.info("Loading ground truth excel from: %s", EXCEL_PATH)
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        logger.exception("Error reading Ground Truth Excel file: %s", EXCEL_PATH)
        raise RuntimeError(f"Error reading Excel file {EXCEL_PATH}: {e}")

    df = df.fillna("")

    return df


def get_ground_truth(question):
    """
    Return the Ground Truth answer for a question.
    """

    df = load_ground_truth()

    for _, row in df.iterrows():

        sample_question = str(row["Sample_query"]).strip()

        if sample_question.lower() == question.strip().lower():

            logger.info("Ground truth match found for question: %s -> candidate: %s", question, row.get("Name", ""))
            return {
                "candidate_name": row["Name"],
                "ground_truth": row["GroundTruth_Answer"]
            }

    logger.info("No ground truth found for question: %s", question)
    return None