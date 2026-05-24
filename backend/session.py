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
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)

# Nazwa pliku z metadanymi sesji zapisywanego w katalogu danych sesji.
# Pozwala odtworzyc sesje (stan, urzadzenia, wyniki) po restarcie backendu.
META_FILENAME = "session.json"


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
    is_camera: bool = True
    joined_at: float = field(default_factory=time.time)
    ws_connected: bool = False
    calib_frame_count: int = 0
    capture_frame_count: int = 0


@dataclass
class CalibResult:
    reproj_error: float     # RMS stereo [px]
    params_path: str        # sciezka do stereo.json
    rms_left: float = 0.0   # RMS kalibracji lewej kamery [px]
    rms_right: float = 0.0  # RMS kalibracji prawej kamery [px]
    warning: str | None = None  # ostrzezenie diagnostyczne (np. baza nie-pozioma)
    computed_at: float = field(default_factory=time.time)


@dataclass
class MeasResult:
    width_mm: float
    length_mm: float
    height_mm: float
    volume_voxel_mm3: float   # objetosc metoda kolumnowa (height-field)
    volume_bbox_mm3: float    # objetosc bounding box (W*L*H)
    volume_hull_mm3: float | None  # objetosc convex hull (None gdy niedostepna)
    fill_ratio: float         # voxel / bbox - "pelnosc" bryly [0..1]
    validation_passed: bool
    pallet_rms_mm: float
    n_object_pts: int
    n_pallet_inliers: int
    issues: list[str]
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
        self._lock = asyncio.Lock()  # per-session lock for concurrent mutations

    # --- Sciezki -----------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.data_root / self.session_id

    @property
    def meta_path(self) -> Path:
        return self.data_dir / META_FILENAME

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
        return len(self.devices) >= 10

    def left_camera(self) -> Optional["Device"]:
        """Lewa kamera stereo: is_camera=True, lider pierwszeński, potem kolejność dołączenia."""
        cams = sorted(
            (d for d in self.devices.values() if d.is_camera),
            key=lambda d: (not d.is_leader, d.joined_at),
        )
        return cams[0] if cams else None

    def right_camera(self) -> Optional["Device"]:
        """Prawa kamera stereo: druga kamera wg tej samej kolejności co left_camera."""
        cams = sorted(
            (d for d in self.devices.values() if d.is_camera),
            key=lambda d: (not d.is_leader, d.joined_at),
        )
        return cams[1] if len(cams) >= 2 else None

    def min_calib_frames(self) -> int:
        """Minimalna liczba klatek kalibracyjnych wsrod urzadzen-kamer."""
        cams = [d for d in self.devices.values() if d.is_camera]
        if not cams:
            return 0
        return min(d.calib_frame_count for d in cams)

    def min_capture_frames(self) -> int:
        cams = [d for d in self.devices.values() if d.is_camera]
        if not cams:
            return 0
        return min(d.capture_frame_count for d in cams)

    # --- Serializacja (persystencja na dysku) ------------------------------

    def to_dict(self) -> dict:
        """Reprezentacja sesji do zapisu w session.json."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "devices": {did: asdict(d) for did, d in self.devices.items()},
            "calib_result": asdict(self.calib_result) if self.calib_result else None,
            "meas_result": asdict(self.meas_result) if self.meas_result else None,
        }

    @classmethod
    def from_dict(cls, data: dict, data_root: Path) -> "Session":
        """Odtwarza sesje z session.json. Stan polaczenia WS jest transientowy
        i zawsze resetowany do False (po restarcie nie ma aktywnych polaczen)."""
        session = cls(data["session_id"], data_root)
        session.state = SessionState(data["state"])
        session.created_at = data.get("created_at", time.time())

        for did, dev in data.get("devices", {}).items():
            dev = dict(dev)
            dev["ws_connected"] = False  # transient - brak aktywnych polaczen po wczytaniu
            session.devices[did] = Device(**dev)

        if data.get("calib_result"):
            session.calib_result = CalibResult(**data["calib_result"])
        if data.get("meas_result"):
            session.meas_result = MeasResult(**data["meas_result"])

        return session


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
        self._load_all()

    # --- Persystencja ------------------------------------------------------

    def _load_all(self) -> None:
        """Wczytuje wszystkie zapisane sesje z dysku przy starcie.

        Skanuje data_root w poszukiwaniu plikow session.json. Dzieki temu
        po restarcie backendu uzytkownicy moga wrocic do swoich sesji
        wraz z zapisanymi danymi (kalibracja, wyniki pomiaru, obrazy).
        """
        for meta in sorted(self.data_root.glob(f"*/{META_FILENAME}")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                session = Session.from_dict(data, self.data_root)
                self._sessions[session.session_id] = session
            except Exception as exc:
                log.warning("Nie udalo sie wczytac sesji z %s: %s", meta, exc)
        if self._sessions:
            log.info("Wczytano %d zapisanych sesji z dysku", len(self._sessions))

    def _write_meta(self, session: Session) -> None:
        """Zapisuje metadane sesji do session.json (best-effort)."""
        try:
            session.data_dir.mkdir(parents=True, exist_ok=True)
            session.meta_path.write_text(
                json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Nie udalo sie zapisac meta sesji %s: %s", session.session_id, exc)

    async def save(self, session_id: str) -> None:
        """Utrwala biezacy stan sesji na dysku."""
        session = self._sessions.get(session_id)
        if session:
            self._write_meta(session)

    def save_sync(self, session_id: str) -> None:
        """Wersja synchroniczna save() - do uzycia w watku roboczym."""
        session = self._sessions.get(session_id)
        if session:
            self._write_meta(session)

    # --- CRUD --------------------------------------------------------------

    async def create(self) -> Session:
        session_id = uuid.uuid4().hex[:8]
        session = Session(session_id, self.data_root)
        session.data_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            self._sessions[session_id] = session
        self._write_meta(session)
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
        """Ustawia stan sesji (bez walidacji przejsc - route handler sprawdza).

        Utrwala pelne metadane sesji (w tym ewentualne wyniki ustawione tuz
        przed zmiana stanu, np. calib_result / meas_result w tasks.py)."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.state = state
                self._write_meta(session)


# Singleton importowany przez reszte modulow
store = SessionStore()
