import os

# Disable local model path and bot security key for all tests
os.environ["LISA_LOCAL_MODEL_PATH"] = ""
os.environ["LISA_BOT_SECURITY_KEY"] = ""
