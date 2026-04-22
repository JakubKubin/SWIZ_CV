# backend/session.py
"""Zarzadzanie stanem sesji w pamieci operacyjnej.

Sesja reprezentuje pojedynczą sesje pomiarową z dwoma urządzeniami
(lewa + prawa kamera). Dane sa trzymane w pamieci; na dysku zapisywane sa
jedynie obrazy i pliki wynikowe JSON.

Maszyna stanow:
  IDLE → CALIBRATING → READY → PROCESSING → DONE
  Blad kalibracji: CALIBRATING → IDLE  (mozliwy retry)
  Blad pomiaru:    PROCESSING  → READY (mozliwy retry)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import config


class SessionState(str, Enum):
    IDLE        = "IDLE"
    CALIBRATING = "CALIBRATING"
    READY       = "READY"
    PROCESSING  = "PROCESSING"
    DONE        = "DONE"


# ---------------------------------------------------------------------------
# Dataclassy wewnetrzne
# ---------------------------------------------------------------------------

@dataclass
class Device:
    device_id: str
    mac: str
    is_leader: bool
    joined_at: float = field(default_factory=time.time)
    ws_connected: bool = False
    calib_frame_count: int = 0
    capture_frame_count: int = 0


@dataclass
class CalibResult:
    reproj_error: float
    params_path: str        # sciezka do stereo.json
    computed_at: float = field(default_factory=time.time)


@dataclass
class MeasResult:
    width_mm: float
    length_mm: float
    height_mm: float
    validation_passed: bool
    pallet_rms_mm: float
    n_object_pts: int
    n_pallet_inliers: int
    issues: list
    report: str
    measured_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Sesja
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, session_id: str, data_root: Path):
        self.session_id = session_id
        self.state = SessionState.IDLE
        self.devices: dict[str, Device] = {}
        self.created_at = time.time()
        self.calib_result: Optional[CalibResult] = None
        self.meas_result: Optional[MeasResult] = None
        self.data_root = data_root

    # --- Sciezki -----------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.data_root / self.session_id

    def calib_dir(self, device_id: str) -> Path:
        return self.data_dir / "calib" / device_id

    def capture_dir(self, device_id: str) -> Path:
        return self.data_dir / "captures" / device_id

    # --- Pomocniki ---------------------------------------------------------

    def leader(self) -> Optional[Device]:
        return next((d for d in self.devices.values() if d.is_leader), None)

    def follower(self) -> Optional[Device]:
        return next((d for d in self.devices.values() if not d.is_leader), None)

    def is_full(self) -> bool:
        return len(self.devices) >= 2

    def min_calib_frames(self) -> int:
        """Minimalna liczba klatek kalibracyjnych wsrod urzadzen."""
        if not self.devices:
            return 0
        return min(d.calib_frame_count for d in self.devices.values())

    def min_capture_frames(self) -> int:
        if not self.devices:
            return 0
        return min(d.capture_frame_count for d in self.devices.values())

    # --- Serializacja do odpowiedzi ----------------------------------------

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state,
            "devices": [
                {
                    "device_id": d.device_id,
                    "mac": d.mac,
                    "is_leader": d.is_leader,
                    "joined_at": d.joined_at,
                    "ws_connected": d.ws_connected,
                    "calib_frame_count": d.calib_frame_count,
                    "capture_frame_count": d.capture_frame_count,
                }
                for d in self.devices.values()
            ],
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Globalny magazyn sesji
# ---------------------------------------------------------------------------

class SessionStore:
    """Bezpieczny (asyncio) magazyn sesji w pamieci."""

    def __init__(self, data_root: str = "data"):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    # --- CRUD --------------------------------------------------------------

    async def create(self) -> Session:
        session_id = uuid.uuid4().hex[:8]
        session = Session(session_id, self.data_root)
        session.data_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            self._sessions[session_id] = session
        return session

    async def get(self, session_id: str) -> Session:
        """Zwraca sesje lub rzuca KeyError."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def get_sync(self, session_id: str) -> Session:
        """Wersja synchroniczna - do uzycia w watku roboczym (tasks.py)."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        import shutil
        shutil.rmtree(session.data_dir, ignore_errors=True)
        return True

    async def list_all(self) -> list[Session]:
        return list(self._sessions.values())

    # --- Stan sesji --------------------------------------------------------

    async def set_state(self, session_id: str, state: SessionState) -> None:
        """Ustawia stan sesji (bez walidacji przejsc - route handler sprawdza)."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.state = state

    def set_state_sync(self, session_id: str, state: SessionState) -> None:
        """Wersja synchroniczna - do uzycia w watku roboczym."""
        session = self._sessions.get(session_id)
        if session:
            session.state = state


# Singleton importowany przez reszte modulow
store = SessionStore()
