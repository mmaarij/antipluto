#!/bin/bash

echo "Installing Anti-PLUTO with all optional dependencies..."
pip install -e ".[classifier,api,dev]"

echo "Downloading spaCy models..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl

echo "Setup complete! Note: GPU acceleration requires manual PyTorch CUDA installation."
