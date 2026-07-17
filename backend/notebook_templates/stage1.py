"""Stage 1 (non-instruction fine-tuning) notebook template."""

from backend.notebook_templates._helpers import code, lora_target_modules_literal, markdown, notebook


def build_stage1_notebook(domain: str, model_id: str, lora: dict, learning_rate: float, epochs: int) -> dict:
    target_modules = lora_target_modules_literal(lora["target_modules"])

    cells = [
        markdown([
            "# Stage 1: Non-Instruction Fine-Tuning\n",
            "\n",
            f"This notebook adapts `{model_id}` to {domain} vocabulary and tone by "
            "training directly on raw domain text (no question/answer structure yet "
            "-- that comes in Stage 2). Run this on Google Colab with a T4 GPU: "
            "Runtime > Change runtime type > T4 GPU.",
        ]),
        code([
            "%%capture\n",
            "!pip install unsloth\n",
            "!pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git",
        ]),
        markdown([
            "## 1. Load the raw domain text\n",
            "\n",
            "Upload `data/non_instruction_data.txt` to this Colab session (or clone "
            "the repo), then read it in. Paragraphs are separated by blank lines.",
        ]),
        code([
            'raw_text = open("data/non_instruction_data.txt", encoding="utf-8").read()\n',
            'paragraphs = [p.strip() for p in raw_text.split("\\n\\n") if p.strip()]\n',
            'print(f"Loaded {len(paragraphs)} paragraphs")\n',
            "print(paragraphs[0])",
        ]),
        markdown([
            "## 2. Clean and chunk the text\n",
            "\n",
            "Each paragraph is already a short, self-contained chunk, so no further "
            "splitting is needed. We just wrap each one into a HuggingFace `Dataset` "
            "with a single `text` column.",
        ]),
        code([
            "from datasets import Dataset\n",
            "\n",
            'dataset = Dataset.from_dict({"text": paragraphs})\n',
            "dataset",
        ]),
        markdown(["## 3. Load the base model using Unsloth"]),
        code([
            "from unsloth import FastLanguageModel\n",
            "import torch\n",
            "\n",
            "max_seq_length = 2048\n",
            "\n",
            "model, tokenizer = FastLanguageModel.from_pretrained(\n",
            f'    model_name = "{model_id}",\n',
            "    max_seq_length = max_seq_length,\n",
            "    dtype = None,\n",
            "    load_in_4bit = True,\n",
            ")",
        ]),
        markdown([
            "## 4. Apply LoRA (QLoRA, since the base model is loaded in 4-bit)\n",
            "\n",
            "`r` is the LoRA rank -- how large the small trainable adapter matrices "
            "are. `lora_alpha` scales how strongly the adapter's updates affect the "
            "model. `target_modules` lists which layers get an adapter -- here, all "
            "the attention and MLP projection layers.",
        ]),
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
            "## 5. Train on the raw text\n",
            "\n",
            "This is plain next-token-prediction training on the `text` column -- "
            "the same objective used for the original pretraining, just on our "
            f"{domain} paragraphs instead of the internet.",
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
            "        packing = True,\n",
            "        per_device_train_batch_size = 2,\n",
            "        gradient_accumulation_steps = 4,\n",
            f"        num_train_epochs = {epochs},\n",
            f"        learning_rate = {learning_rate},\n",
            "        fp16 = not is_bfloat16_supported(),\n",
            "        bf16 = is_bfloat16_supported(),\n",
            "        logging_steps = 5,\n",
            '        output_dir = "stage1_outputs",\n',
            '        optim = "adamw_8bit",\n',
            "        seed = 3407,\n",
            "    ),\n",
            ")\n",
            "trainer_stats = trainer.train()",
        ]),
        markdown(["## 6. Save the adapter"]),
        code([
            'model.save_pretrained("stage1_adapter")\n',
            'tokenizer.save_pretrained("stage1_adapter")',
        ]),
        markdown([
            "## 7. Test the model after non-instruction fine-tuning\n",
            "\n",
            f"We only expect the model to sound more {domain}-flavored here, not "
            "yet answer questions well -- instruction-following comes in Stage 2.",
        ]),
        code([
            "FastLanguageModel.for_inference(model)\n",
            'inputs = tokenizer([paragraphs[0][:30]], return_tensors="pt").to("cuda")\n',
            "outputs = model.generate(**inputs, max_new_tokens=64)\n",
            'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        ]),
    ]
    return notebook(cells)
