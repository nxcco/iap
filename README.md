# Anfängerpraktikum

This repository contains experiments and implementations for a series of numerical computing projects. Each subproject is an independent Python project with its own virtual environment and Python version managed by `pyenv`.

## Subprojects

- **iterative-refinement** — Implements the iterative refinement method applied to the 1D Poisson problem. Checks convergence and plots results.
- **approximate-multiplication** — Runs statistical experiments comparing approximate multiplication to exact multiplication, including an error distribution heatmap.
- **approximate-iterative-refinement** — Combines approximate multiplication with iterative refinement.

## HPC Submodule

The `HPC` directory is a git submodule provided by ZITI. Access is restricted to authorized members due to copyright.

## Setup

Each subproject uses `pyenv` to manage its Python version. To set one up, navigate into the subproject directory and run:

```bash
pyenv install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
