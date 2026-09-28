#!/usr/bin/env bash
set -e
# Setup Python env and install dependencies.  Override MIXMIL_REF when the
# submitted job should use a review branch containing local MixMIL changes.
MIXMIL_REF="${MIXMIL_REF:-main}"
pip install pandas matplotlib scipy scikit-learn seaborn scanpy anndata statannotations fastparquet torch-scatter \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html \
    "git+https://github.com/AIH-SGML/mixmil.git@${MIXMIL_REF}"
python -c "import torch; print(torch.__version__)"


echo "Running: python run.py $@"
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python run.py "$@"
