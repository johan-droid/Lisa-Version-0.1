import os
import re

def audit():
    print("Running Security Audit on d:\\Lisa Version 1...")
    patterns = {
        "Hardcoded Secrets": re.compile(r'(api_key|secret|password|token)\s*=\s*["\'][a-zA-Z0-9_\-]{20,}["\']', re.IGNORECASE),
        "Ungated Eval/Exec": re.compile(r'\b(eval|exec)\b'),
    }
    
    issues_found = 0
    for root, dirs, files in os.walk("lisa"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                content = open(path, "r", encoding="utf-8", errors="ignore").read()
                for name, pattern in patterns.items():
                    for match in pattern.finditer(content):
                        # Filter out expected occurrences or comments
                        line_no = content[:match.start()].count("\n") + 1
                        print(f"[{name}] {path}:{line_no} - Match: {match.group(0)[:50]}")
                        issues_found += 1
                        
    if issues_found == 0:
        print("SECURITY AUDIT: PASS")
    else:
        print(f"SECURITY AUDIT: WARNING ({issues_found} potential issues found)")

if __name__ == "__main__":
    audit()
