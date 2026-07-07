from difflib import SequenceMatcher

from excel_reader import get_ground_truth


def calculate_similarity(answer1, answer2):
    """
    Calculate similarity percentage.
    """

    similarity = SequenceMatcher(
        None,
        answer1.lower(),
        answer2.lower()
    ).ratio()

    return round(similarity * 100, 2)


def evaluate_answer(question, generated_answer):
    """
    Compare generated answer with Ground Truth.
    """

    gt = get_ground_truth(question)

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