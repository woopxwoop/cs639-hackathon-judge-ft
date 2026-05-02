FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml ./
COPY src/ src/

# torch is already in the base image; install remaining deps without reinstalling torch
RUN uv pip install --system --no-deps \
        typer rich datasets pandas scikit-learn peft trl \
        unsloth unsloth_zoo && \
    uv pip install --system -e .

ENTRYPOINT ["ft"]
