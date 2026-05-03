from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from hackathon_judge_ft.config import ENABLE_THINKING

SAMPLING_PARAMS = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 8192}

VERDICT_RE = re.compile(r"VERDICT\s*:\s*(A|B|TIE)\b", re.IGNORECASE)


def parse_verdict(response: str) -> str | None:
    if "</think>" in response:
        response = response.split("</think>")[-1]
    last = None
    for m in VERDICT_RE.finditer(response):
        last = m
    if last is None:
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
    model_name: str = "unsloth/Qwen3.5-4B",
    model_tag: str = "Qwen/Qwen3.5-4B-sft",
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
            enable_thinking=ENABLE_THINKING,
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
    n_total = len(test_dataset)

    for i, ex in enumerate(test_dataset):
        prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
        response, predicted, n_prompt, n_completion, latency, finish_reason = run_inference(prompt_messages)
        verdict = predicted if predicted is not None else "invalid"
        frontier_match = verdict == ex["answer"]
        frontier_correct += int(frontier_match)
        print(f"  [{i+1}/{n_total}] {ex['pair_id'][:8]} pos={ex['position']}  {verdict} vs {ex['answer']}  {'✓' if frontier_match else '✗'}  ({latency:.1f}s)", flush=True)
        rows.append({
            "messages":          list(prompt_messages) + [{"role": "assistant", "content": response}],
            "judgment_id":       ex["judgment_id"],
            "pair_id":           ex["pair_id"],
            "hackathon":         ex["hackathon"],
            "position":          ex["position"],
            "project_a_id":      ex["project_a_id"],
            "project_b_id":      ex["project_b_id"],
            "verdict":           verdict,
            "frontier_verdict":  ex["answer"],
            "gt_a_result":       ex["gt_a_result"],
            "gt_b_result":       ex["gt_b_result"],
            "model":             model_tag,
            "prompt_tokens":     n_prompt,
            "completion_tokens": n_completion,
            "finish_reason":     finish_reason,
            "latency_s":         latency,
            "sampling":          json.dumps(SAMPLING_PARAMS),
            "frontier_match":    frontier_match,
        })

    n_consistent = 0
    n_checked = 0
    pair_results: dict[str, dict] = {}
    for row in rows:
        pair_results.setdefault(row["pair_id"], {})[row["position"]] = row["verdict"]

    for pair_id in test_pairs:
        ab_v = pair_results.get(pair_id, {}).get("ab")
        ba_v = pair_results.get(pair_id, {}).get("ba")
        if ab_v is None or ba_v is None:
            continue
        consistent = (ab_v == "A" and ba_v == "B") or (ab_v == "B" and ba_v == "A")
        n_consistent += int(consistent)
        n_checked += 1

    return {
        "frontier_agreement":  frontier_correct / len(rows) if rows else 0,
        "position_consistency": n_consistent / n_checked if n_checked else 0,
        "n_test":  len(rows),
        "n_pairs": n_checked,
        "rows":    rows,
    }


def save_parquet(rows: list[dict], output: Path) -> None:
    parquet_rows = [{k: v for k, v in r.items() if k not in ("frontier_match", "frontier_verdict")} for r in rows]
    table = pa.Table.from_pylist(parquet_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)
