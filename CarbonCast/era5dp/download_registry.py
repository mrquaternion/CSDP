"""Persistent download presence registry for cluster-aware checks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


CHECK_MODE_ENV = "ERA5DP_DOWNLOAD_CHECK_MODE"
REGISTRY_FILE_ENV = "ERA5DP_DOWNLOAD_REGISTRY_FILE"

MODE_LOCAL = "local"
MODE_HYBRID = "hybrid"
MODE_REGISTRY = "registry"
VALID_MODES = {MODE_LOCAL, MODE_HYBRID, MODE_REGISTRY}


def _to_pattern(path: Path, kind: str):
    if kind == "nc":
        return "*.nc"
    if kind == "tif":
        return "*.tif"
    return "*"


class DownloadRegistry:
    """Tracks dataset availability independently from transient local files."""

    def __init__(self):
        raw_mode = os.environ.get(CHECK_MODE_ENV, MODE_LOCAL).strip().lower()
        self.mode = raw_mode if raw_mode in VALID_MODES else MODE_LOCAL

        registry_path = os.environ.get(REGISTRY_FILE_ENV, "").strip()
        self.registry_path = Path(registry_path).expanduser().resolve() if registry_path else None
        self.entries: dict[str, dict[str, bool]] = {}
        self._load()

    @property
    def use_registry(self) -> bool:
        return self.mode in {MODE_HYBRID, MODE_REGISTRY}

    @property
    def use_local(self) -> bool:
        return self.mode in {MODE_LOCAL, MODE_HYBRID}

    def has_data(self, path: str | Path, kind: str = "any") -> bool:
        resolved = Path(path).resolve()
        key = str(resolved)

        if self.use_local and self._has_local_data(resolved, kind):
            self.mark_available(resolved, kind)
            return True

        if self.use_registry:
            return self._entry_matches(self.entries.get(key, {}), kind)

        return False

    def mark_available(self, path: str | Path, kind: str = "any"):
        if not self.use_registry:
            return

        key = str(Path(path).resolve())
        entry = self.entries.setdefault(key, {"any": False, "nc": False, "tif": False})

        entry["any"] = True
        if kind == "nc":
            entry["nc"] = True
        elif kind == "tif":
            entry["tif"] = True

        self._save()

    def _has_local_data(self, path: Path, kind: str) -> bool:
        if not path.is_dir():
            return False

        pattern = _to_pattern(path, kind)
        return any(path.rglob(pattern))

    @staticmethod
    def _entry_matches(entry: dict[str, bool], kind: str) -> bool:
        if not entry:
            return False
        if kind == "nc":
            return bool(entry.get("nc") or entry.get("any"))
        if kind == "tif":
            return bool(entry.get("tif") or entry.get("any"))
        return bool(entry.get("any") or entry.get("nc") or entry.get("tif"))

    def _load(self):
        if not self.registry_path or not self.registry_path.exists():
            return

        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        entries = payload.get("entries")
        if isinstance(entries, dict):
            for key, value in entries.items():
                if isinstance(value, dict):
                    self.entries[key] = {
                        "any": bool(value.get("any")),
                        "nc": bool(value.get("nc")),
                        "tif": bool(value.get("tif")),
                    }

    def _save(self):
        if not self.registry_path:
            return

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "entries": self.entries,
        }
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
