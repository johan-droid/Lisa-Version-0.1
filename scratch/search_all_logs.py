import os
import json
import re

brain_dir = r"C:\Users\sahoo\.gemini\antigravity\brain"
for root, dirs, files in os.walk(brain_dir):
    for file in files:
        if file == "overview.txt":
            filepath = os.path.join(root, file)
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        content = data.get("content", "")
                        if not content:
                            continue
                        # search for pattern: user ID or chat ID or a large number (9-10 digits) or bot username
                        # Or phrases containing "id" or "bot" or "ping"
                        content_lower = content.lower()
                        if (
                            "chat" in content_lower
                            or "bot" in content_lower
                            or "telegram" in content_lower
                            or "id" in content_lower
                        ):
                            # Let's print lines matching these that look like numbers or IDs
                            found_nums = re.findall(r"\b\d{8,15}\b", content)
                            if (
                                found_nums
                                or "ping" in content_lower
                                or "bot id" in content_lower
                                or "chat id" in content_lower
                            ):
                                print(
                                    f"File: {os.path.basename(os.path.dirname(root))} - Line: {idx} - Step: {data.get('step_index')}"
                                )
                                print(f"  Nums: {found_nums}")
                                print(f"  Content: {content[:300]}")
                                print("-" * 50)
                    except Exception:
                        pass
