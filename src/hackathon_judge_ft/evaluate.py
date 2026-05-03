from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset

SAMPLING_PARAMS = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 8192}


VERDICT_RE = re.compile(r"VERDICT\s*:\s*(A|B|TIE)\b", re.IGNORECASE)


def parse_verdict(response: str) -> str | None:
    if "</think>" in response:
        response = response.split("</think>")[-1]
    last = None
    for m in VERDICT_RE.finditer(response):
        last = m
    if last is None:
        # fallback for truncated outputs
        if re.search(r"\bProject\s+A\b", response, re.IGNORECASE):
            return "A"
        if re.search(r"\bProject\s+B\b", response, re.IGNORECASE):
            return "B"
        return None
    v = last.group(1).upper()
    return "tie" if v == "TIE" else v


def run(
    test_dataset: Dataset,
    test_pairs: set,
    adapter_path: str,
    model_name: str = "unsloth/Qwen3-4B",
    model_tag: str = "Qwen/Qwen3-4B-sft",
) -> dict:
    import torch
    from unsloth import FastModel
    from peft import PeftModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    max_new_tokens = SAMPLING_PARAMS["max_tokens"]

    def run_inference(prompt_messages):
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        n_prompt = inputs.input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=SAMPLING_PARAMS["temperature"],
                top_p=SAMPLING_PARAMS["top_p"],
                do_sample=True,
            )
        latency = time.perf_counter() - t0

        n_completion = outputs.shape[1] - n_prompt
        finish_reason = "length" if n_completion >= max_new_tokens else "stop"
        response = tokenizer.decode(outputs[0][n_prompt:], skip_special_tokens=True)
        return response, parse_verdict(response), n_prompt, n_completion, latency, finish_reason

    rows = []
    frontier_correct = 0

    for ex in test_dataset:
        prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
        response, predicted, n_prompt, n_completion, latency, finish_reason = run_inference(prompt_messages)

        verdict = predicted if predicted is not None else "invalid"
        frontier_match = verdict == ex["answer"]
        frontier_correct += int(frontier_match)

        full_messages = list(prompt_messages) + [{"role": "assistant", "content": response}]

        rows.append({
            "messages":         full_messages,
            "judgment_id":      ex["judgment_id"],
            "pair_id":          ex["pair_id"],
            "hackathon":        ex["hackathon"],
            "position":         ex["position"],
            "project_a_id":     ex["project_a_id"],
            "project_b_id":     ex["project_b_id"],
            "verdict":          verdict,
            "gt_a_result":      ex["gt_a_result"],
            "gt_b_result":      ex["gt_b_result"],
            "model":            model_tag,
            "prompt_tokens":    n_prompt,
            "completion_tokens": n_completion,
            "finish_reason":    finish_reason,
            "latency_s":        latency,
            "sampling":         json.dumps(SAMPLING_PARAMS),
            "frontier_match":   frontier_match,
        })

    results_df = pd.DataFrame(rows)

    n_consistent = 0
    n_checked = 0
    for pair_id in test_pairs:
        pair_rows = results_df[results_df["pair_id"] == pair_id]
        ab = pair_rows[pair_rows["position"] == "ab"]
        ba = pair_rows[pair_rows["position"] == "ba"]
        if ab.empty or ba.empty:
            continue
        ab_v = ab.iloc[0]["verdict"]
        ba_v = ba.iloc[0]["verdict"]
        consistent = (ab_v == "A" and ba_v == "B") or (ab_v == "B" and ba_v == "A")
        n_consistent += int(consistent)
        n_checked += 1

    return {
        "frontier_agreement": frontier_correct / len(test_dataset) if test_dataset else 0,
        "position_consistency": n_consistent / n_checked if n_checked else 0,
        "n_test": len(test_dataset),
        "n_pairs": n_checked,
        "results_df": results_df,
    }


def save_parquet(results_df: pd.DataFrame, output: Path) -> None:
    # Drop internal-only column before saving
    out_df = results_df.drop(columns=["frontier_match"])
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)
