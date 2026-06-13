import json

filepath = r"C:\Users\sahoo\.gemini\antigravity\brain\0ac1dd40-0c09-4a16-97d0-01e7d18c6b7e\.system_generated\logs\overview.txt"
with open(filepath, encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            content = data.get("content", "")
            if data.get("source") == "USER_EXPLICIT":
                print(f"User Request (Step {data.get('step_index')}):")
                print(content)
                print("=" * 40)
        except Exception:
            pass
