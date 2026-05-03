from __future__ import annotations

import torch
from datasets import Dataset


def run(
    train_dataset: Dataset,
    model_name: str = "unsloth/Qwen3-4B",
    output_dir: str = "./hackathon_judge_lora",
    epochs: int = 3,
    r: int = 32,
    max_seq_length: int = 8192,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
) -> None:
    from unsloth import FastModel
    from trl import SFTConfig, SFTTrainer

    # unsloth_zoo generates Linear_peft_forward.py that references VARIANT_KWARG_KEYS
    # from peft's module scope but forgets to import it; inject it after Unsloth has patched imports.
    try:
        import builtins
        from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
        builtins.VARIANT_KWARG_KEYS = VARIANT_KWARG_KEYS
    except ImportError:
        pass

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        r=r,
        lora_alpha=r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_tokenized = train_dataset.map(preprocess, num_proc=1)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        save_strategy="epoch",
        gradient_checkpointing=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_tokenized,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
