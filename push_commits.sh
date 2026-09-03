#!/bin/bash
set -e

echo "=== Commit 4: Streamlit config and gitignore ==="
git add .streamlit/ .gitignore
git commit -m "config(streamlit): add .streamlit/config.toml and update .gitignore"

echo "=== Commit 5: Deployment files ==="
git add Dockerfile .dockerignore render.yaml pyproject.toml uv.lock push_commits.sh
git commit -m "deploy: update Dockerfile to py3.11, add render.yaml, .dockerignore, pyproject.toml"

echo "=== Pushing all commits ==="
git push origin main

echo "Done! All commits pushed successfully."
