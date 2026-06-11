import os
import pickle
import hmac
import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("lisa.snapshot")

def get_snapshot_path(settings: Any) -> Path:
    # Resolve relative to workspace_root
    root = getattr(settings, "workspace_root", None)
    if root:
        return Path(root) / "data" / "lisa_state.snap"
    return Path("data/lisa_state.snap")

def get_hmac_key(settings: Any) -> bytes:
    # Use bot_security_key if available, fallback to a persistent key or random bytes
    key_str = getattr(settings, "bot_security_key", None)
    if key_str:
        return key_str.encode("utf-8")
    return b"lisa_default_secret_key_change_me_or_set_security_key"

def save_snapshot(state_data: Any, settings: Any) -> bool:
    try:
        snapshot_file = get_snapshot_path(settings)
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        key = get_hmac_key(settings)
        
        # Serialize data
        serialized = pickle.dumps(state_data)
        
        # Calculate HMAC
        mac = hmac.new(key, serialized, hashlib.sha256).digest()
        
        # Write [HMAC (32 bytes)] + [Serialized Data]
        with open(snapshot_file, "wb") as f:
            f.write(mac)
            f.write(serialized)
            
        logger.info(f"Saved state snapshot to {snapshot_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save state snapshot: {e}")
        return False

def load_snapshot(settings: Any) -> Any:
    snapshot_file = get_snapshot_path(settings)
    if not snapshot_file.exists():
        return None
        
    try:
        key = get_hmac_key(settings)
        with open(snapshot_file, "rb") as f:
            mac = f.read(32)
            serialized = f.read()
            
        # Verify HMAC
        expected_mac = hmac.new(key, serialized, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, expected_mac):
            logger.warning("Snapshot integrity check failed! Tampering suspected. Ignoring snapshot.")
            return None
            
        # Deserialize data
        state_data = pickle.loads(serialized)
        logger.info(f"Successfully loaded state snapshot from {snapshot_file}")
        return state_data
    except Exception as e:
        logger.error(f"Failed to load state snapshot: {e}")
        return None

def clear_snapshot(settings: Any) -> None:
    snapshot_file = get_snapshot_path(settings)
    if snapshot_file.exists():
        try:
            os.remove(snapshot_file)
            logger.info(f"Cleared state snapshot file at {snapshot_file}.")
        except Exception as e:
            logger.warning(f"Could not remove snapshot file: {e}")
