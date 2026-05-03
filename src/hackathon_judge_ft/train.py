from __future__ import annotations

import math

import torch
from datasets import Dataset

from hackathon_judge_ft.config import ENABLE_THINKING


class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template: str) -> None:
        self.tokenizer = tokenizer
        template_ids = tokenizer(text=response_template, add_special_tokens=False)[
            "input_ids"
        ]
        self.response_token_ids = (
            template_ids[0]
            if template_ids and isinstance(template_ids[0], list)
            else template_ids
        )

    @staticmethod
    def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
        last_start = len(sequence) - len(subsequence)
        for start in range(last_start + 1):
            if sequence[start : start + len(subsequence)] == subsequence:
                return start
        return -1

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        texts = [example["text"] for example in examples]
        batch = self.tokenizer(
            text=texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        for i, input_ids in enumerate(batch["input_ids"].tolist()):
            response_start = self._find_subsequence(input_ids, self.response_token_ids)
            if response_start < 0:
                raise ValueError(
                    "response template not found in tokenized training example"
                )
            response_content_start = response_start + len(self.response_token_ids)
            labels[i, :response_content_start] = -100
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


def run(
    train_dataset: Dataset,
    model_name: str = "unsloth/Qwen3.5-4B",
    output_dir: str = "./hackathon_judge_lora",
    epochs: int = 3,
    r: int = 32,
    max_seq_length: int = 12288,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 42,
    num_proc: int = 8,
) -> None:
    from transformers import Trainer, TrainingArguments

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    peft_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=ENABLE_THINKING,
        )
        return {
            "text": text,
            "n_tokens": len(
                tokenizer(text=text, add_special_tokens=False)["input_ids"]
            ),
        }

    train_tokenized = train_dataset.map(preprocess, num_proc=num_proc)
    n_before_filter = len(train_tokenized)
    train_tokenized = train_tokenized.filter(lambda r: r["n_tokens"] <= max_seq_length)
    n_dropped = n_before_filter - len(train_tokenized)
    if n_dropped:
        print(
            f"  dropped {n_dropped} training examples longer than {max_seq_length} tokens"
        )
    train_tokenized = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c != "text"]
    )

    steps_per_epoch = math.ceil(
        len(train_tokenized) / (batch_size * gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.03))
    print(f"  warmup: {warmup_steps} steps (3% of {total_steps} total steps)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        data_seed=seed,
        num_train_epochs=epochs,
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        use_liger_kernel=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        data_collator=CompletionOnlyCollator(
            tokenizer, response_template="<|im_start|>assistant\n"
        ),
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
