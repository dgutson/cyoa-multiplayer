#!/bin/bash

shellcheck lint.sh
pymarkdown scan README.md
yamllint ./*.yml
pylint main.py
ruff check main.py
mypy --strict main.py
validate-pyproject pyproject.toml
radon cc --show-complexity .|grep -v "A (1)"|grep -v "A (2)"|grep -v "A (3)"|grep -v "A (4)"
yapf -d main.py
