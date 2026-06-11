import os
import shutil
import hashlib
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("lisa.evolution_guard")

ERROR_TRACKER: dict[str, list[float]] = {}  # skill_name -> list of error timestamps

def get_file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def write_journal_entry(event_type: str, payload: dict[str, Any], backup_dir: Path) -> None:
    try:
        journal_file = backup_dir.parent / "data" / "evolution_journal.jsonl"
        journal_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload
        }
        with open(journal_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write evolution journal entry: {e}")

def snapshot_skills_dir(skills_dir: Path, backup_dir: Path) -> Path | None:
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_name = backup_dir / f"skills_snapshot_{timestamp}"
        
        # Create zip archive of skills_dir
        archive_path_str = shutil.make_archive(
            base_name=str(archive_name),
            format="zip",
            root_dir=str(skills_dir)
        )
        archive_path = Path(archive_path_str)
        logger.info(f"Created skills directory snapshot at {archive_path}")
        return archive_path
    except Exception as e:
        logger.error(f"Failed to snapshot skills directory: {e}")
        return None

def restore_skills_snapshot(archive_path: Path, skills_dir: Path) -> bool:
    try:
        if not archive_path.exists():
            logger.error(f"Snapshot archive does not exist: {archive_path}")
            return False
            
        # Clear all files in skills_dir and extract archive
        for item in skills_dir.iterdir():
            if item.is_file():
                os.remove(item)
            elif item.is_dir():
                shutil.rmtree(item)
                
        shutil.unpack_archive(str(archive_path), str(skills_dir))
        logger.info(f"Restored skills directory from snapshot: {archive_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to restore skills directory: {e}")
        return False

def record_skill_deployment(skill_name: str, skill_path: Path, snapshot_path: Path | None, backup_dir: Path) -> None:
    checksum = get_file_sha256(skill_path)
    write_journal_entry("skill_deployed", {
        "skill_name": skill_name,
        "skill_path": str(skill_path),
        "checksum": checksum,
        "snapshot_path": str(snapshot_path) if snapshot_path else None
    }, backup_dir)

def track_skill_error(skill_name: str, error_msg: str, skills_dir: Path, backup_dir: Path) -> bool:
    """
    Tracks an error for a given skill.
    If the skill has 3+ errors in the last 1 hour, triggers auto-rollback to the latest snapshot
    that was taken before deployment, and returns True. Otherwise returns False.
    """
    now = time.time()
    if skill_name not in ERROR_TRACKER:
        ERROR_TRACKER[skill_name] = []
    ERROR_TRACKER[skill_name].append(now)
    
    # Filter errors in the last 1 hour (3600 seconds)
    one_hour_ago = now - 3600
    ERROR_TRACKER[skill_name] = [t for t in ERROR_TRACKER[skill_name] if t > one_hour_ago]
    
    write_journal_entry("skill_error", {
        "skill_name": skill_name,
        "error": error_msg,
        "error_count_last_hour": len(ERROR_TRACKER[skill_name])
    }, backup_dir)
    
    if len(ERROR_TRACKER[skill_name]) >= 3:
        logger.warning(f"Skill '{skill_name}' triggered {len(ERROR_TRACKER[skill_name])} errors in the last hour. Triggering auto-rollback.")
        return rollback_to_latest_snapshot(skill_name, skills_dir, backup_dir)
        
    return False

def rollback_to_latest_snapshot(skill_name: str, skills_dir: Path, backup_dir: Path) -> bool:
    snapshot_path: Path | None = None
    journal_file = backup_dir.parent / "data" / "evolution_journal.jsonl"
    
    if journal_file.exists():
        try:
            with open(journal_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines):
                entry = json.loads(line.strip())
                if entry.get("event_type") == "skill_deployed" and entry.get("skill_name") == skill_name:
                    snap = entry.get("snapshot_path")
                    if snap:
                        snapshot_path = Path(snap)
                        break
        except Exception as e:
            logger.error(f"Error searching journal for rollback snapshot: {e}")

    # If no specific snapshot path is found in the journal, find the latest zip in backup_dir
    if not snapshot_path or not snapshot_path.exists():
        logger.info("No specific snapshot found in journal for skill, looking for latest zip in backup directory.")
        zips = sorted(backup_dir.glob("skills_snapshot_*.zip"), key=os.path.getmtime)
        if zips:
            snapshot_path = zips[-1]

    if snapshot_path and snapshot_path.exists():
        logger.info(f"Rolling back to snapshot: {snapshot_path}")
        success = restore_skills_snapshot(snapshot_path, skills_dir)
        if success:
            write_journal_entry("rollback", {
                "skill_name": skill_name,
                "reason": "Too many errors (3+ in 1 hour)",
                "snapshot_path": str(snapshot_path)
            }, backup_dir)
            # Reset error tracker for this skill
            ERROR_TRACKER[skill_name] = []
            return True
    else:
        logger.error("No valid snapshot found to rollback to.")
        write_journal_entry("rollback_failed", {
            "skill_name": skill_name,
            "reason": "No valid snapshot found"
        }, backup_dir)
        
    return False
