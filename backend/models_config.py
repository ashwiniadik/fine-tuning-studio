"""Curated base models and their fine-tuning defaults for Fine-Tuning Studio.

Hyperparameter defaults (LoRA config, learning rates, epochs) reuse the exact
values validated in finance-faq-assistant-finetuning's three notebooks.
"""

DEFAULT_LORA = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

DEFAULT_STAGE_HPARAMS = {
    "stage1": {"learning_rate": 2e-4, "epochs": 2},
    "stage2": {"learning_rate": 2e-4, "epochs": 3},
    "stage3": {"learning_rate": 5e-6, "epochs": 2},
}

MODELS = {
    "qwen2.5-0.5b": {
        "display_name": "Qwen2.5 0.5B",
        "unsloth_model_id": "unsloth/Qwen2.5-0.5B",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
    "qwen2.5-1.5b": {
        "display_name": "Qwen2.5 1.5B",
        "unsloth_model_id": "unsloth/Qwen2.5-1.5B",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
    "tinyllama-1.1b": {
        "display_name": "TinyLlama 1.1B",
        "unsloth_model_id": "unsloth/tinyllama-bnb-4bit",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
    "llama-3.2-1b": {
        "display_name": "Llama 3.2 1B",
        "unsloth_model_id": "unsloth/Llama-3.2-1B",
        "lora": DEFAULT_LORA,
        **DEFAULT_STAGE_HPARAMS,
    },
}
