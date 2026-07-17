from backend.models_config import MODELS

REQUIRED_STAGE_KEYS = {"learning_rate", "epochs"}
REQUIRED_LORA_KEYS = {"r", "lora_alpha", "lora_dropout", "target_modules"}


def test_expected_models_present():
    assert set(MODELS.keys()) == {
        "qwen2.5-0.5b",
        "qwen2.5-1.5b",
        "tinyllama-1.1b",
        "llama-3.2-1b",
    }


def test_all_models_have_required_fields():
    for key, cfg in MODELS.items():
        assert "display_name" in cfg, key
        assert cfg["unsloth_model_id"].startswith("unsloth/"), key
        assert set(cfg["lora"].keys()) == REQUIRED_LORA_KEYS, key
        for stage in ["stage1", "stage2", "stage3"]:
            assert stage in cfg, f"{key} missing {stage}"
            assert set(cfg[stage].keys()) == REQUIRED_STAGE_KEYS, f"{key} {stage}"
