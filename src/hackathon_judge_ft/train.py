from __future__ import annotations

import torch
from datasets import Dataset

from hackathon_judge_ft.config import ENABLE_THINKING


def run(
    train_dataset: Dataset,
    model_name: str = "unsloth/Qwen3.5-4B",
    output_dir: str = "./hackathon_judge_lora",
    epochs: int = 3,
    r: int = 32,
    max_seq_length: int = 16384,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-4,
    seed: int = 42,
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
    )

    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=ENABLE_THINKING,
        )
        return {
            "n_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
        }

    train_tokenized = train_dataset.map(preprocess, num_proc=1)
    n_before_filter = len(train_tokenized)
    train_tokenized = train_tokenized.filter(lambda r: r["n_tokens"] <= max_seq_length)
    n_dropped = n_before_filter - len(train_tokenized)
    if n_dropped:
        print(f"  dropped {n_dropped} training examples longer than {max_seq_length} tokens")

    training_args = SFTConfig(
        output_dir=output_dir,
        seed=seed,
        data_seed=seed,
        num_train_epochs=epochs,
        save_strategy="epoch",
        gradient_checkpointing=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=5,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        assistant_only_loss=True,
        max_length=max_seq_length,
        dataset_kwargs={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
