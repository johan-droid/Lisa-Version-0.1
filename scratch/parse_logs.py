import json

with open(
    r"C:\Users\sahoo\.gemini\antigravity\brain\0ac1dd40-0c09-4a16-97d0-01e7d18c6b7e\.system_generated\logs\overview.txt",
    encoding="utf-8",
) as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            if data.get("source") == "USER_EXPLICIT" or "content" in data:
                content = data.get("content", "")
                if (
                    "ping" in content.lower()
                    or "bot" in content.lower()
                    or "telegram" in content.lower()
                    or "id" in content.lower()
                    or "8746" in content.lower()
                ):
                    print(f"--- Step {data.get('step_index')} (Line {idx}) ---")
                    print(content)
        except Exception as e:
            print(f"Error parsing line {idx}: {e}")
