import os

brain_dir = r"C:\Users\sahoo\.gemini\antigravity\brain"
for root, dirs, files in os.walk(brain_dir):
    for file in files:
        filepath = os.path.join(root, file)
        # only inspect files under brain, skip huge ones if any, but search text/json/md
        if not file.endswith((".txt", ".json", ".md", ".yaml", ".yml")):
            continue
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "8746921085" in content or "LisaVersion1_bot" in content:
                    print(f"Found in: {filepath}")
                    # Print matching line numbers and contents
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if "8746921085" in line or "LisaVersion1_bot" in line:
                            print(f"  Line {idx+1}: {line[:200]}")
        except Exception:
            pass
