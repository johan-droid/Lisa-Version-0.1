import os
import re

workspace_dir = r"d:\Lisa Version 1"
for root, dirs, files in os.walk(workspace_dir):
    # prune directories
    dirs[:] = [
        d for d in dirs if d not in (".git", ".venv", "node_modules", "__pycache__")
    ]
    for file in files:
        filepath = os.path.join(root, file)
        if file.endswith(
            (".py", ".json", ".yaml", ".yml", ".txt", ".md", ".local", ".example")
        ):
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    # Look for 9-10 digit numbers
                    matches = re.findall(r"\b\d{9,10}\b", content)
                    if matches:
                        print(f"Found in {filepath}: {matches}")
            except Exception:
                pass
