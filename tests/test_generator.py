from backend.generator import select_stages


def test_instruction_only():
    assert select_stages(has_raw_text=False, has_preference=False) == ["stage2"]


def test_raw_text_and_instruction():
    assert select_stages(has_raw_text=True, has_preference=False) == ["stage1", "stage2"]


def test_instruction_and_preference():
    assert select_stages(has_raw_text=False, has_preference=True) == ["stage2", "stage3"]


def test_all_three():
    assert select_stages(has_raw_text=True, has_preference=True) == ["stage1", "stage2", "stage3"]
