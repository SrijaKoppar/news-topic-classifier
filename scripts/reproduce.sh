#!/usr/bin/env bash
# Rebuild everything from a fresh clone. Run from the repo root with the venv active.
set -euo pipefail
python -m pytest -q
python -m src.explore
python -m src.check_splits
python -m src.train --model all
python -m src.evaluate select
python -m src.evaluate test --force   # --force: results are committed, but models/final/ is not
python -m src.abstain
python -m src.abstain_diagnostic
python -m scripts.smoke_test_api
echo "Done. Now run: git diff --stat results/   (empty, or tiny float differences, expected)"