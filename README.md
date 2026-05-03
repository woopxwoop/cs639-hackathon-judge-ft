# Hackathon Judge Fine-tuning

Fine-tunes Qwen3-4B via SFT to imitate Qwen3.5-27B pairwise hackathon judgments.

## Setup

```bash
uv sync
```

## Usage

```bash
# Train on all hackathons (1 epoch)
ft train

# Train on a single hackathon
ft train --hackathon treehacks-2026

# More options
ft train --hackathon treehacks-2026 --epochs 2 --output ./my_adapter

# Evaluate a saved adapter
ft evaluate ./hackathon_judge_lora
ft evaluate ./hackathon_judge_lora --hackathon treehacks-2026

# Save judgments as parquet (same schema as twangodev/devpost-hacks-judgments)
ft evaluate ./hackathon_judge_lora --output ./judgments_sft.parquet

# Evaluate per-epoch checkpoints
ft evaluate ./hackathon_judge_lora/checkpoint-500 --output ./judgments_epoch1.parquet
ft evaluate ./hackathon_judge_lora/checkpoint-1000 --output ./judgments_epoch2.parquet
```

Use the same `--hackathon`, `--test-size`, and `--seed` for both `train` and `evaluate`.

## Docker

```bash
docker compose build
HF_TOKEN=<token> docker compose run --rm train
HF_TOKEN=<token> ADAPTER=./hackathon_judge_lora docker compose run --rm evaluate

# Override compose defaults by passing ft subcommands without the leading `ft`
HF_TOKEN=<token> docker compose run --rm train train --hackathon treehacks-2026 --epochs 2
HF_TOKEN=<token> docker compose run --rm evaluate evaluate ./hackathon_judge_lora --hackathon treehacks-2026
```

## Dataset

[`twangodev/devpost-hacks-judgments`](https://huggingface.co/datasets/twangodev/devpost-hacks-judgments) — pairwise judgments from Qwen3.5-27B across 8 hackathons.

Available hackathon configs: `cal-hacks-12-0`, `treehacks-2026`, `treehacks-2025`, `treehacks-2024`, `hackgt-12`, `madhacks-fall-2025`, `madhacks`, `pennapps-xxv`
