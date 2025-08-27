# Copilot Project Instructions

## 1. Overview

This file enables AI coding assistants to generate features that are fully aligned with the architecture, style, and conventions of the PROPAGATOR wildfire simulation project. All guidance is based strictly on observed patterns and actual code in the repository—no invented or external best practices are included.

## 2. File Category Reference

Below are the file categories present in this project, with representative examples and unique conventions:

### core-engine
- **What it is:** The main simulation logic, including the cellular automata engine, model functions, and supporting utilities.
- **Examples:**
  - `src/propagator/propagator.py`
  - `src/propagator/models.py`
- **Key conventions:**
  - All wildfire spread logic is grid-based and uses numpy arrays and dataclasses.
  - Model functions are modular and pluggable.
  - No global mutable state except for a project-wide RNG.

### cli
- **What it is:** Command-line interface logic for running and configuring simulations.
- **Examples:**
  - `src/propagator_cli/cli.py`
  - `src/propagator_cli/args_parser.py`
- **Key conventions:**
  - Pydantic models are used for CLI argument parsing and validation.
  - Console output is styled with the `rich` library.
  - No GUI or web interface is present.

### io-configuration
- **What it is:** Configuration and parameter management for simulations.
- **Examples:**
  - `src/propagator_io/configuration.py`
- **Key conventions:**
  - All config is handled via Pydantic models with strict field validation.
  - No ad-hoc or global config patterns.

### io-loader
- **What it is:** Modules for loading geospatial input data (e.g., DEM, fuel, ignitions).
- **Examples:**
  - `src/propagator_io/loader/geotiff.py`
- **Key conventions:**
  - Loader classes use standard geospatial libraries and protocols for extensibility.

### io-writer
- **What it is:** Modules for writing simulation outputs to geospatial formats.
- **Examples:**
  - `src/propagator_io/writer/geojson.py`
- **Key conventions:**
  - Writer classes output only standard formats (GeoTIFF, GeoJSON).
  - Protocols are used for extensibility and type safety.

### tests
- **What it is:** Automated tests for simulation logic and CLI.
- **Examples:**
  - `tests/test_propagator.py`
- **Key conventions:**
  - All tests use pytest, with fixtures and mocks for simulation components.

### examples
- **What it is:** Example script, and data for demonstrating usage.
- **Examples:**
  - `src/example.py`
- **Key conventions:**
  - Example is a minimal Python script.
  - Example data is always in the `example/` directory.

### docs
- **What it is:** Project documentation, including API reference and usage guides.
- **Examples:**
  - `docs/index.md`
  - `docs/cli.md`
- **Key conventions:**
  - All docs are Markdown, built with MkDocs and mkdocstrings.

### config
- **What it is:** Project and documentation configuration files.
- **Examples:**
  - `pyproject.toml`
  - `mkdocs.yml`
- **Key conventions:**
  - All config is TOML or YAML, with minimal formatting and no `setup.py`.

### data
- **What it is:** Example and lookup data for simulations.
- **Examples:**
  - `example/dem_clip.tif`
  - `example/prob_table.txt`
- **Key conventions:**
  - All spatial data is GeoTIFF or GeoJSON; tables are plain text.

### ci-cd
- **What it is:** Continuous integration and deployment automation.
- **Examples:**
  - `.github/workflows/docs.yaml`
- **Key conventions:**
  - Only GitHub Actions workflows are used, focused on docs build/deploy.

### site-generated
- **What it is:** MkDocs-generated static site files.
- **Examples:**
  - `site/index.html`
- **Key conventions:**
  - All files in `site/` are generated and never edited manually.

## 3. Feature Scaffold Guide

When planning a new feature:
- Determine which categories of files are needed (e.g., core-engine for new simulation logic, io-loader for new data sources, tests for coverage).
- Place new files in the appropriate directory (e.g., `src/propagator/` for core logic, `src/propagator_io/loader/` for loaders, `tests/` for tests).
- Follow naming and structure conventions from the style guides above.
- For example, a new simulation model would require:
  - A new function in `src/propagator/functions.py`
  - Updates to dataclasses in `src/propagator/models.py`
  - Tests in `tests/test_propagator.py`

## 4. Integration Rules

- All wildfire spread logic must use the grid-based, numpy-driven cellular automata approach.
- All CLI features must be exposed via Pydantic-based argument parsing and styled with `rich`.
- All input/output must use standard geospatial formats and libraries (rasterio, geopandas, pyproj, shapely).
- All configuration must be added as Pydantic models/settings.
- All tests must use pytest conventions.
- All documentation must be Markdown and compatible with MkDocs.
- No GUI, web, or non-cellular automata fire models are allowed.

## 5. Example Prompt Usage

> "Add a new vegetation type and corresponding probability model to the simulation."

Copilot would respond with:
- `src/propagator/constants.py` (add new vegetation type constant)
- `src/propagator/functions.py` (add new probability model function)
- `src/propagator/models.py` (update dataclasses if needed)
- `tests/test_propagator.py` (add tests for new model)
- `docs/reference/propagator.md` (document the new model)

All new files and changes must follow the conventions and integration rules described above.
