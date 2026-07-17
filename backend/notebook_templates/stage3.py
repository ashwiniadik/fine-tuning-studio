"""Stage 3 (DPO preference alignment) notebook template."""

from backend.notebook_templates._helpers import alpaca_prompt_text, code, lora_target_modules_literal, markdown, notebook


def build_stage3_notebook(
    domain: str,
    model_id: str,
    lora: dict,
    learning_rate: float,
    epochs: int,
    previous_adapter: str,
) -> dict:
    target_modules = lora_target_modules_literal(lora["target_modules"])
    prompt_text = alpaca_prompt_text(domain)

    cells = [
        markdown([
            "# Stage 3: DPO Preference Alignment\n",
            "\n",
            f"Continues from the Stage 2 SFT adapter and uses the preference "
            f"dataset to teach the model to prefer specific, helpful {domain} "
            "answers over generic ones. Run on Google Colab with a T4 GPU.",
        ]),
        code([
            "%%capture\n",
            "!pip install unsloth trl\n",
            "!pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git",
        ]),
        markdown([
            "## 1. Load the SFT model\n",
            "\n",
            f"Upload the `{previous_adapter}` folder from Stage 2, plus "
            "`data/preference_dataset.jsonl`. We load the SFT adapter twice: once "
            "as the trainable model DPO will update, and once more as a frozen "
            "reference copy -- DPO measures how far the trainable model's "
            "preferences have shifted relative to this frozen SFT reference, so "
            "the reference must actually be the SFT model, not the original "
            "pretrained base model.",
        ]),
        code([
            "from unsloth import FastLanguageModel\n",
            "\n",
            "max_seq_length = 2048\n",
            "\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{previous_adapter}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")\n",
            "model = FastLanguageModel.get_peft_model(\n",
            "    model,\n",
            f'    r = {lora["r"]},\n',
            f"    target_modules = [{target_modules}],\n",
            f'    lora_alpha = {lora["lora_alpha"]},\n',
            f'    lora_dropout = {lora["lora_dropout"]},\n',
            '    bias = "none",\n',
            '    use_gradient_checkpointing = "unsloth",\n',
            "    random_state = 3407,\n",
            ")\n",
            "\n",
            "# A second, separate copy of the SFT model, loaded without a fresh\n",
            "# LoRA adapter attached, so it stays frozen and serves as the DPO reference.\n",
            "ref_model, _ = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{previous_adapter}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")",
        ]),
        markdown([
            "## 2. Load and format the preference dataset\n",
            "\n",
            "The file already has `prompt`, `chosen`, and `rejected` columns, "
            "which is exactly what `DPOTrainer` expects. The `chosen`/`rejected` "
            "responses can be used as-is, but `prompt` needs to be wrapped in the "
            "same Alpaca-style template used for Stage 2 SFT training -- the "
            "model was trained to expect questions in that format, so DPO must "
            "compare responses under the same format it will actually be asked "
            "at inference time.",
        ]),
        code([
            "from datasets import load_dataset\n",
            "\n",
            f'ALPACA_PROMPT = """{prompt_text}"""\n',
            "\n",
            "def format_dpo_prompt(example):\n",
            '    example["prompt"] = ALPACA_PROMPT.format(example["prompt"], "")\n',
            "    return example\n",
            "\n",
            'dpo_dataset = load_dataset("json", data_files="data/preference_dataset.jsonl", split="train")\n',
            "dpo_dataset = dpo_dataset.map(format_dpo_prompt)\n",
            "dpo_dataset[0]",
        ]),
        markdown([
            "## 3. Configure and run DPO training\n",
            "\n",
            "`beta` controls how strongly the model is pushed toward the chosen "
            "response versus how close it stays to the reference (SFT) model. "
            "`learning_rate` is much lower here than in Stages 1-2 because DPO "
            "only needs to nudge an already-competent model's preferences, not "
            "relearn from scratch. We pass our separately-loaded `ref_model` "
            "explicitly, rather than `ref_model=None`, so the reference is "
            "genuinely the Stage 2 SFT model.",
        ]),
        code([
            "from trl import DPOTrainer, DPOConfig\n",
            "from unsloth import is_bfloat16_supported\n",
            "\n",
            "dpo_trainer = DPOTrainer(\n",
            "    model = model,\n",
            "    ref_model = ref_model,\n",
            "    args = DPOConfig(\n",
            "        max_length = max_seq_length,\n",
            "        per_device_train_batch_size = 2,\n",
            "        gradient_accumulation_steps = 4,\n",
            f"        num_train_epochs = {epochs},\n",
            f"        learning_rate = {learning_rate},\n",
            "        beta = 0.1,\n",
            "        fp16 = not is_bfloat16_supported(),\n",
            "        bf16 = is_bfloat16_supported(),\n",
            "        logging_steps = 5,\n",
            '        output_dir = "stage3_outputs",\n',
            '        optim = "adamw_8bit",\n',
            "        seed = 3407,\n",
            "    ),\n",
            "    train_dataset = dpo_dataset,\n",
            "    processing_class = tokenizer,\n",
            ")\n",
            "dpo_trainer.train()",
        ]),
        markdown(["## 4. Save the DPO-aligned model"]),
        code([
            'model.save_pretrained("stage3_model")\n',
            'tokenizer.save_pretrained("stage3_model")',
        ]),
        markdown(["## 5. Test the model after DPO"]),
        code([
            "import json as _json\n",
            "\n",
            'first_prompt = _json.loads(open("data/preference_dataset.jsonl").readline())["prompt"]\n',
            "\n",
            "FastLanguageModel.for_inference(model)\n",
            'prompt = ALPACA_PROMPT.format(first_prompt, "")\n',
            'inputs = tokenizer([prompt], return_tensors="pt").to("cuda")\n',
            "outputs = model.generate(**inputs, max_new_tokens=128)\n",
            'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        ]),
    ]
    return notebook(cells)
