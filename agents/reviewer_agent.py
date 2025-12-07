# agents/reviewer_agent.py

from typing import Dict, List, Any


class ReviewerAgent:
    """
    Takes shots + QC issues and produces human-readable review comments.
    Later, this can call Gemini to propose JSON edits.
    """

    def __init__(self):
        pass

    def review(self, shots: List[Dict[str, Any]], qc_report: Dict[str, List[str]]) -> Dict[str, Any]:
            """
            For now, just wraps QC report in a more narrative structure.
            """
            summary_comments: List[str] = []

            if not qc_report:
                summary_comments.append("All shots passed basic QC checks.")
            else:
                summary_comments.append(
                    f"{len(qc_report)} shot(s) have potential issues. Review suggested."
                )

            # In the future: suggest JSON patches per shot based on qc_report.

            return {
                "summary": summary_comments,
                "per_shot_issues": qc_report,
            }