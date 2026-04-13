#!/usr/bin/env bash
set -e

pip install -r requirements.txt

# Pre-download the embedding model during build so it's cached at runtime
python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"
