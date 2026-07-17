"""Ties together validation, stage selection, and notebook generation."""


def select_stages(has_raw_text: bool, has_preference: bool) -> list[str]:
    """Instruction data (Stage 2) is always required and always included.
    Raw text (Stage 1) and preference data (Stage 3) are optional add-ons."""
    stages = []
    if has_raw_text:
        stages.append("stage1")
    stages.append("stage2")
    if has_preference:
        stages.append("stage3")
    return stages
