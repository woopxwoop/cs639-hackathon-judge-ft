from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Fine-tune and evaluate a small Qwen3.5 model as a hackathon judge.",
)

console = Console()


@app.command()
def train(
    hackathon: Optional[str] = typer.Option(
        None, "--hackathon", "-h",
        help="Single hackathon config (e.g. treehacks-2026). Default: all hackathons.",
    ),
    model: str = typer.Option("unsloth/Qwen3.5-4B", "--model"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    r: int = typer.Option(32, "--r", help="LoRA rank"),
    batch_size: int = typer.Option(32, "--batch-size"),
    gradient_accumulation_steps: int = typer.Option(1, "--gradient-accumulation-steps"),
    learning_rate: float = typer.Option(3e-4, "--learning-rate", "--lr"),
    max_seq_length: int = typer.Option(16384, "--max-seq-length"),
    output: Path = typer.Option(Path("./hackathon_judge_lora"), "--output", "-o"),
    test_size: float = typer.Option(0.2, "--test-size"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Load frontier judgments, fine-tune Qwen3.5 with LoRA, save adapter weights."""
    from hackathon_judge_ft import data as data_mod
    from hackathon_judge_ft import train as train_mod

    console.print(f"Loading dataset ({hackathon or 'all hackathons'})...")
    ds = data_mod.load_frontier(hackathon)
    data_mod.validate(ds)
    console.print(f"  {len(ds)} frontier rows loaded")

    train_ds, _, train_pairs, _ = data_mod.split(ds, test_size=test_size, seed=seed)
    console.print(f"  train: {len(train_ds)} examples ({len(train_pairs)} unique pairs)")

    console.print(
        f"Training {model} | LoRA r={r} | batch={batch_size} x grad_accum={gradient_accumulation_steps} | lr={learning_rate:g} | {epochs} epoch(s)..."
    )
    train_mod.run(
        train_ds,
        model_name=model,
        output_dir=str(output),
        epochs=epochs,
        r=r,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
    )

    typer.echo(f"saved adapter → {output}")


@app.command()
def evaluate(
    adapter: Path = typer.Argument(..., help="Path to saved LoRA adapter"),
    hackathon: Optional[str] = typer.Option(
        None, "--hackathon", "-h",
        help="Single hackathon config (must match what was used during train).",
    ),
    model: str = typer.Option("unsloth/Qwen3.5-4B", "--model"),
    model_tag: str = typer.Option("Qwen/Qwen3.5-4B-sft", "--model-tag", help="Model name written to output parquet"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save judgments parquet to this path"),
    test_size: float = typer.Option(0.2, "--test-size"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Evaluate a saved adapter: frontier agreement + position bias consistency."""
    from hackathon_judge_ft import data as data_mod
    from hackathon_judge_ft import evaluate as eval_mod

    console.print(f"Loading dataset ({hackathon or 'all hackathons'})...")
    ds = data_mod.load_frontier(hackathon)
    _, test_ds, _, test_pairs = data_mod.split(ds, test_size=test_size, seed=seed)
    console.print(f"  test: {len(test_ds)} examples ({len(test_pairs)} unique pairs)")

    console.print(f"Running inference with {model} + adapter {adapter}...")
    metrics = eval_mod.run(test_ds, test_pairs, str(adapter.resolve()), model_name=model, model_tag=model_tag)

    typer.echo(f"frontier agreement:   {metrics['frontier_agreement']*100:.1f}%  ({metrics['n_test']} examples)")
    typer.echo(f"position consistency: {metrics['position_consistency']*100:.0f}%  ({metrics['n_pairs']} pairs)")

    if output:
        eval_mod.save_parquet(metrics["rows"], output)
        typer.echo(f"saved judgments → {output}")
