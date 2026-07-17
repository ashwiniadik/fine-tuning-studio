"""Stage 2 (instruction fine-tuning / SFT) notebook template."""

from backend.notebook_templates._helpers import alpaca_prompt_text, code, lora_target_modules_literal, markdown, notebook


def build_stage2_notebook(
    domain: str,
    model_id: str,
    lora: dict,
    learning_rate: float,
    epochs: int,
    previous_adapter: str | None,
) -> dict:
    target_modules = lora_target_modules_literal(lora["target_modules"])
    load_model_name = previous_adapter if previous_adapter else model_id
    prompt_text = alpaca_prompt_text(domain)

    intro = (
        f"Continues from the Stage 1 adapter and teaches the model to answer "
        f"{domain} questions directly, using the instruction/response dataset."
        if previous_adapter
        else f"Teaches `{model_id}` to answer {domain} questions directly, using "
        "the instruction/response dataset."
    )
    upload_note = (
        f"Upload the `{previous_adapter}` folder from Stage 1, plus "
        "`data/instruction_dataset.jsonl`."
        if previous_adapter
        else "Upload `data/instruction_dataset.jsonl` to this Colab session (or "
        "clone the repo)."
    )

    cells = [
        markdown([
            "# Stage 2: Instruction Fine-Tuning (SFT)\n",
            "\n",
            f"{intro} Run on Google Colab with a T4 GPU.",
        ]),
        code([
            "%%capture\n",
            "!pip install unsloth\n",
            "!pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git",
        ]),
        markdown([
            "## 1. Load tokenizer and the model\n",
            "\n",
            upload_note,
        ]),
        code([
            "from unsloth import FastLanguageModel\n",
            "import torch\n",
            "\n",
            "max_seq_length = 2048\n",
            "\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{load_model_name}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")",
        ]),
        markdown([
            "## 2. Load and format the instruction dataset\n",
            "\n",
            "Each `{instruction, response}` pair is wrapped in a simple Alpaca-style "
            "prompt template so the model learns the exact format it will be asked "
            "to answer in at inference time.",
        ]),
        code([
            "from datasets import load_dataset\n",
            "\n",
            f'ALPACA_PROMPT = """{prompt_text}"""\n',
            "\n",
            "EOS_TOKEN = tokenizer.eos_token\n",
            "\n",
            "def formatting_prompts_func(examples):\n",
            "    texts = []\n",
            '    for instruction, response in zip(examples["instruction"], examples["response"]):\n',
            "        texts.append(ALPACA_PROMPT.format(instruction, response) + EOS_TOKEN)\n",
            '    return {"text": texts}\n',
            "\n",
            'dataset = load_dataset("json", data_files="data/instruction_dataset.jsonl", split="train")\n',
            "dataset = dataset.map(formatting_prompts_func, batched=True)\n",
            'dataset[0]["text"]',
        ]),
        markdown(["## 3. Apply LoRA"]),
        code([
            "model = FastLanguageModel.get_peft_model(\n",
            "    model,\n",
            f'    r = {lora["r"]},\n',
            f"    target_modules = [{target_modules}],\n",
            f'    lora_alpha = {lora["lora_alpha"]},\n',
            f'    lora_dropout = {lora["lora_dropout"]},\n',
            '    bias = "none",\n',
            '    use_gradient_checkpointing = "unsloth",\n',
            "    random_state = 3407,\n",
            ")",
        ]),
        markdown([
            "## 4. Train the model\n",
            "\n",
            "`packing=False` here because each instruction/response pair needs to "
            "stay its own training example with a clean boundary -- packing "
            "multiple short examples into one sequence would blur those "
            "boundaries and give far fewer real training steps.",
        ]),
        code([
            "from trl import SFTTrainer, SFTConfig\n",
            "from unsloth import is_bfloat16_supported\n",
            "\n",
            "trainer = SFTTrainer(\n",
            "    model = model,\n",
            "    processing_class = tokenizer,\n",
            "    train_dataset = dataset,\n",
            "    args = SFTConfig(\n",
            '        dataset_text_field = "text",\n',
            "        max_length = max_seq_length,\n",
            "        packing = False,\n",
            "        per_device_train_batch_size = 2,\n",
            "        gradient_accumulation_steps = 4,\n",
            f"        num_train_epochs = {epochs},\n",
            f"        learning_rate = {learning_rate},\n",
            "        fp16 = not is_bfloat16_supported(),\n",
            "        bf16 = is_bfloat16_supported(),\n",
            "        logging_steps = 5,\n",
            '        output_dir = "stage2_outputs",\n',
            '        optim = "adamw_8bit",\n',
            "        seed = 3407,\n",
            "    ),\n",
            ")\n",
            "trainer_stats = trainer.train()",
        ]),
        markdown(["## 5. Save the adapter"]),
        code([
            'model.save_pretrained("stage2_adapter")\n',
            'tokenizer.save_pretrained("stage2_adapter")',
        ]),
        markdown(["## 6. Run inference after training"]),
        code([
            "import json as _json\n",
            "\n",
            'first_question = _json.loads(open("data/instruction_dataset.jsonl").readline())["instruction"]\n',
            "\n",
            "FastLanguageModel.for_inference(model)\n",
            'prompt = ALPACA_PROMPT.format(first_question, "")\n',
            'inputs = tokenizer([prompt], return_tensors="pt").to("cuda")\n',
            "outputs = model.generate(**inputs, max_new_tokens=128)\n",
            'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        ]),
    ]
    return notebook(cells)
