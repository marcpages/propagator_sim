# Repository Guidelines

## Project Structure & Module Organization
- `propagator/`: Core simulation code (e.g., `propagator.py`, `scheduler.py`, `functions.py`).
- `propagator_io/`: Input/Output utilities and data models (`configuration.py`, `input.py`, `output.py`).
- `propagator_cli/`: CLI and console helpers (`cli.py`, `args_parser.py`, `console.py`).
- `example/`: Sample params and assets for quick runs.
- `tests/`: Pytest suite (e.g., `tests/test_propagator.py`).
- Entrypoints: `main.py` (current CLI flow), `_main.py` (legacy).

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install (pip): `pip install -e .`  (dev: `pip install -e .[dev]`)
- Install (uv): `uv sync`  (uses `pyproject.toml`/`uv.lock`)
- Run tests: `pytest -q`
- Lint: `ruff check .`
- Format: `ruff format .`
- Quick run: `python main.py` or `python main.py --help`
- Example run: `python main.py -f ./example/params.json -of ./example/output -tl 24 -dem ./example/dem.tif -veg ./example/veg.tif`

## Coding Style & Naming Conventions
- Python ≥ 3.13, 4‑space indentation, type hints preferred.
- Use Ruff for lint and format; keep imports sorted and unused code removed.
- Modules: `snake_case.py`; classes: `CamelCase`; functions/vars: `snake_case`.
- Keep public dataclasses and Pydantic models documented with concise docstrings.

## Testing Guidelines
- Framework: Pytest (configured in `pyproject.toml`).
- Naming: place tests in `tests/`, files as `test_*.py`, functions as `test_*`.
- Scope: add unit tests for new logic (e.g., Propagator transitions, IO helpers). Ensure deterministic RNG in tests via seeding/mocking.
- Run locally with `pytest -q`; target specific tests with `pytest -k name`.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `chore:`) as seen in history.
- PRs: include a clear description, linked issues, test coverage for changes, and CLI/console output snippets when relevant. Note any breaking changes and update examples if flags/params change.

## Tips & Notes
- Large raster assets: keep out of VCS; use the `example/` set for smoke tests.
- Configuration lives in `propagator_io/configuration.py`; prefer adding new fields there with validation.
