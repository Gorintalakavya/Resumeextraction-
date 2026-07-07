import re

# =====================================================
# Candidate Name Extraction
# =====================================================

def extract_candidate_name(query, candidate_names):
    """
    Extract candidate name from the user's question.

    Example:
        What are Kavya's skills?
        -> Resume_Kavya
    """

    query_lower = query.lower()

    for candidate in candidate_names:

        candidate_lower = candidate.lower()

        # Match complete name
        if candidate_lower in query_lower:
            return candidate

        # Match individual words
        words = candidate_lower.replace("_", " ").split()

        for word in words:

            if len(word) > 2 and word in query_lower:
                return candidate

    return None