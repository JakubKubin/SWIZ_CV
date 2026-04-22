# backend/schemas.py
"""Modele Pydantic dla requestów i odpowiedzi API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sesje
# ---------------------------------------------------------------------------

class JoinRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=64, description="Identyfikator urządzenia")
    mac: str = Field(..., min_length=1, max_length=64, description="Adres MAC urządzenia")
    is_leader: bool = Field(False, description="True = lewa kamera (leader), False = prawa (follower)")


class DeviceOut(BaseModel):
    device_id: str
    mac: str
    is_leader: bool
    joined_at: float
    ws_connected: bool
    calib_frame_count: int
    capture_frame_count: int


class SessionOut(BaseModel):
    session_id: str
    state: str
    devices: list[DeviceOut]
    created_at: float


# ---------------------------------------------------------------------------
# Kalibracja
# ---------------------------------------------------------------------------

class CalibStatusOut(BaseModel):
    state: str
    reproj_error: Optional[float] = None
    message: str


# ---------------------------------------------------------------------------
# Synchronizacja / przechwytywanie
# ---------------------------------------------------------------------------

class TriggerRequest(BaseModel):
    delay_ms: int = Field(500, ge=0, le=10_000, description="Opóźnienie przed przechwyceniem [ms]")


class TriggerOut(BaseModel):
    at: float = Field(description="Unix timestamp momentu przechwycenia")
    delay_ms: int


# ---------------------------------------------------------------------------
# Pomiar
# ---------------------------------------------------------------------------

class MeasurementOut(BaseModel):
    validation_passed: bool
    width_mm: float
    length_mm: float
    height_mm: float
    pallet_rms_mm: float
    n_object_pts: int
    n_pallet_inliers: int
    issues: list[str]
    report: str


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

class HealthOut(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class ErrorOut(BaseModel):
    detail: str
