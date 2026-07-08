from difflib import SequenceMatcher
import sys

from excel_reader import get_ground_truth
from logger import logger


def calculate_similarity(answer1, answer2):
    """
    Calculate similarity percentage.
    """

    try:
        if not answer1 or not answer2:
            return 0.0

        a1 = str(answer1).strip().lower()
        a2 = str(answer2).strip().lower()

        # If the ground-truth text appears verbatim in the generated answer,
        # treat that as a perfect match.
        if a2 and a2 in a1:
            return 100.0

        similarity = SequenceMatcher(None, a1, a2).ratio()
    except Exception as e:
        print(f"Error calculating similarity: {e}", file=sys.stderr)
        return 0.0

    return round(similarity * 100, 2)


def evaluate_answer(question, generated_answer):
    """
    Compare generated answer with Ground Truth.
    """

    try:
        gt = get_ground_truth(question)
    except Exception as e:
        logger.exception("Error loading ground truth for question: %s", question)
        return {
            "candidate_name": "",
            "ground_truth": f"Error loading ground truth: {e}",
            "similarity": 0,
            "status": "ERROR"
        }

    if gt is None:

        return {
            "candidate_name": "",
            "ground_truth": "Question not found in Ground Truth Excel.",
            "similarity": 0,
            "status": "NOT FOUND"
        }

    similarity = calculate_similarity(
        generated_answer,
        gt["ground_truth"]
    )

    logger.info("Calculated similarity for question '%s': %s%%", question, similarity)

    if similarity >= 90:
        status = "Excellent"

    elif similarity >= 75:
        status = "Good"

    elif similarity >= 50:
        status = "Average"

    else:
        status = "Poor"

    return {
        "candidate_name": gt["candidate_name"],
        "ground_truth": gt["ground_truth"],
        "similarity": similarity,
        "status": status
    }
