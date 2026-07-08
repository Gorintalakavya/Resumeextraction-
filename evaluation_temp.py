from difflib import SequenceMatcher
import sys

from excel_reader import get_ground_truth


def calculate_similarity(answer1, answer2):
    """
    Calculate similarity percentage.
    """

    try:
        similarity = SequenceMatcher(
            None,
            answer1.lower(),
            answer2.lower()
        ).ratio()
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
        print(f"Error loading ground truth: {e}", file=sys.stderr)
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
