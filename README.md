# paddling

## Environment

This project uses **[uv](https://docs.astral.sh/uv/)** for package and environment management—**not** `pip` / `python -m pip install`.

- Create/update the venv and install dependencies: `uv sync`
- Run a command in the project environment: `uv run <command>`, e.g. `uv run jupyter lab` or `uv run python -m jupyter lab` to open the TrOCR fine-tuning notebook `main.ipynb`

`pyproject.toml` and `uv.lock` (if present) are the source of truth for dependencies.
