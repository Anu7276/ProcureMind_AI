import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Force offline mock test environment for testing suite
os.environ["LLM_PROVIDER"] = "mock"
