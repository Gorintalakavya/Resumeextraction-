import os
import pandas as pd

from config import GROUND_TRUTH_FILE


EXCEL_PATH = GROUND_TRUTH_FILE


def load_ground_truth():
    """
    Load the Ground Truth Excel file.
    """

    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(
            f"Ground Truth file not found: {EXCEL_PATH}"
        )

    df = pd.read_excel(EXCEL_PATH)

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

            return {
                "candidate_name": row["Name"],
                "ground_truth": row["GroundTruth_Answer"]
            }

    return None