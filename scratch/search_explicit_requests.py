import os
import json
import re

brain_dir = r"C:\Users\sahoo\.gemini\antigravity\brain"
for root, dirs, files in os.walk(brain_dir):
    for file in files:
        if file == "overview.txt":
            filepath = os.path.join(root, file)
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                convo_id = os.path.basename(os.path.dirname(root))
                # Only check directories inside brain
                if not re.match(r"^[0-9a-fA-F\-]{36}$", convo_id):
                    continue
                user_msgs = []
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        if data.get("source") == "USER_EXPLICIT":
                            content = data.get("content", "")
                            if content:
                                user_msgs.append((idx, content))
                    except Exception:
                        pass

                # Check user messages for telegram details
                for idx, content in user_msgs:
                    content_lower = content.lower()
                    if any(
                        w in content_lower
                        for w in [
                            "telegram",
                            "chat_id",
                            "bot_id",
                            "bot token",
                            "ping me",
                            "my chat",
                            "my telegram",
                            "8746",
                        ]
                    ):
                        print(f"Convo ID: {convo_id} - Line: {idx}")
                        print(content)
                        print("-" * 50)
