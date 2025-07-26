#!/bin/bash

shellcheck lint.sh
pymarkdown scan README.md
yamllint ./*.yml
pylint main.py
ruff check main.py
mypy --strict main.py
validate-pyproject pyproject.toml
