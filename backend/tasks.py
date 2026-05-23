# backend/tasks.py
"""Zadania wykonywane w tle (kalibracja, pomiar) + zarzadzanie WebSocket.

Zadania CPU-intensive sa uruchamiane w puli watkow przez asyncio.to_thread(),
zeby nie blokowac petli zdarzen uvicorn.

WebSocketManager zarzadza polaczeniami all devices w ramach sesji.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import cv2

from calibration import calibrate_stereo, save_params, load_params, baseline_warning
from disparity import rectify_pair, compute_disparity
from pointcloud import build_pointcloud, filter_pointcloud, save_ply
from pallet import detect_pallet
from measurement import measure_object, validate_measurement, generate_report
from pipeline import generate_synthetic_scene

from .session import (
    Session, SessionState, CalibResult, MeasResult, store,
)

log = logging.getLogger(__name__)


# ===========================================================================
# WebSocketManager
# ===========================================================================

class WSManager:
    """Utrzymuje aktywne polaczenia WebSocket per (session_id, device_id)."""

    def __init__(self):
        from fastapi import WebSocket
        self._conns: dict[tuple[str, str], Any] = {}
        # tuple[session_id, device_id] -> WebSocket

    async def connect(self, ws: Any, session_id: str, device_id: str) -> None:
        await ws.accept()
        self._conns[(session_id, device_id)] = ws

    def disconnect(self, session_id: str, device_id: str) -> None:
        self._conns.pop((session_id, device_id), None)

    async def send(self, session_id: str, device_id: str, payload: dict) -> None:
        ws = self._conns.get((session_id, device_id))
        if ws is None:
            log.warning("WS send: brak aktywnego polaczenia [%s/%s] dla event=%s",
                        session_id, device_id, payload.get("event"))
            return
        try:
            await ws.send_json(payload)
        except Exception as e:
            log.warning("WS send blad [%s/%s] event=%s: %s",
                        session_id, device_id, payload.get("event"), e)

    async def broadcast(self, session_id: str, payload: dict) -> None:
        """Wysyla wiadomosc do wszystkich urzadzen w sesji."""
        targets = [(did, ws) for (sid, did), ws in self._conns.items() if sid == session_id]
        if not targets:
            log.warning("WS broadcast [%s] event=%s: brak odbiorcow (0 polaczen)",
                        session_id, payload.get("event"))
            return
        results = await asyncio.gather(
            *(ws.send_json(payload) for _, ws in targets),
            return_exceptions=True,
        )
        for (did, _), result in zip(targets, results):
            if isinstance(result, Exception):
                log.warning("WS broadcast blad [%s/%s] event=%s: %s",
                            session_id, did, payload.get("event"), result)


ws_manager = WSManager()


# ===========================================================================
# KALIBRACJA
# ===========================================================================

def _sync_calibrate(session_id: str) -> CalibResult:
    """Synchroniczna kalibracja stereo - uruchamiana w watku roboczym.

    Wczytuje obrazy kalibracyjne z dysku, uruchamia calibrate_stereo()
    i zapisuje wynik do stereo.json.
    """
    session = store.get_sync(session_id)

    leader = session.leader()
    follower = session.follower()
    if leader is None or follower is None:
        raise ValueError("Potrzeba dokladnie 2 urzadzen (leader + follower)")

    left_paths = sorted(session.calib_dir(leader.device_id).glob("frame_*.jpg")) + \
                 sorted(session.calib_dir(leader.device_id).glob("frame_*.png"))
    right_paths = sorted(session.calib_dir(follower.device_id).glob("frame_*.jpg")) + \
                  sorted(session.calib_dir(follower.device_id).glob("frame_*.png"))

    pairs = list(zip(sorted(left_paths), sorted(right_paths)))
    if len(pairs) < 3:
        raise ValueError(
            f"Za malo par kalibracyjnych: {len(pairs)} < 3. "
            f"Leader: {len(left_paths)} kl., Follower: {len(right_paths)} kl."
        )

    log.info("Kalibracja: %d par klatek (%s / %s)", len(pairs),
             leader.device_id, follower.device_id)

    stereo = calibrate_stereo(
        [str(p) for p, _ in pairs],
        [str(p) for _, p in pairs],
    )

    params_path = str(session.data_dir / "stereo.json")
    save_params(stereo, params_path)

    warn = baseline_warning(stereo.T, stereo.reproj_error)
    log.info("Kalibracja OK: stereo=%.4f px (L=%.4f, P=%.4f), params=%s",
             stereo.reproj_error, stereo.left.reproj_error,
             stereo.right.reproj_error, params_path)
    if warn:
        log.warning("[%s] Jakosc kalibracji: %s", session_id, warn)
    return CalibResult(
        reproj_error=float(stereo.reproj_error),
        params_path=params_path,
        rms_left=float(stereo.left.reproj_error),
        rms_right=float(stereo.right.reproj_error),
        warning=warn,
    )


async def calibrate_session(session_id: str) -> None:
    """Async wrapper - uruchamia kalibracje i rozsyla wynik przez WS."""
    try:
        result = await asyncio.to_thread(_sync_calibrate, session_id)

        session = await store.get(session_id)
        session.calib_result = result
        await store.set_state(session_id, SessionState.READY)

        await ws_manager.broadcast(session_id, {
            "event": "calibration_done",
            "reproj_error": result.reproj_error,
            "rms_left": result.rms_left,
            "rms_right": result.rms_right,
            "warning": result.warning,
        })
        log.info("[%s] Kalibracja zakonczona: RMS=%.4f px", session_id, result.reproj_error)

    except Exception as exc:
        log.error("[%s] Blad kalibracji: %s", session_id, exc)
        await store.set_state(session_id, SessionState.IDLE)
        await ws_manager.broadcast(session_id, {
            "event": "error",
            "message": f"Kalibracja nieudana: {exc}",
        })


# ===========================================================================
# POMIAR
# ===========================================================================

def _sync_measure(session_id: str) -> MeasResult:
    """Synchroniczny pelny pipeline pomiarowy - uruchamiany w watku roboczym.

    Kolejne etapy:
      1. Wczytaj kalibracje i obrazy
      2. Rektyfikacja → dysparycja SGBM → glebokos → chmura
      3. Detekcja palety (RANSAC) → segmentacja → pomiar wymiarow
    """
    session = store.get_sync(session_id)

    if session.calib_result is None:
        raise ValueError("Brak kalibracji - najpierw wykonaj kalibracje")

    leader = session.leader()
    follower = session.follower()
    if leader is None or follower is None:
        raise ValueError("Brak urzadzen w sesji")

    stereo = load_params(session.calib_result.params_path, stereo=True)

    # Najnowsze zdjecia pomiarowe
    def latest_image(directory: Path) -> Path:
        imgs = sorted(directory.glob("capture_*.jpg")) + \
               sorted(directory.glob("capture_*.png"))
        if not imgs:
            raise FileNotFoundError(f"Brak zdiec pomiarowych w: {directory}")
        return imgs[-1]

    left_path = latest_image(session.capture_dir(leader.device_id))
    right_path = latest_image(session.capture_dir(follower.device_id))

    log.info("[%s] Pomiar: left=%s right=%s", session_id, left_path.name, right_path.name)

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        raise IOError(f"Nie mozna wczytac zdiec: {left_path}, {right_path}")

    # Etap 2-4: rektyfikacja, dysparycja, glebokos.
    # rectify_pair sam dopasowuje rozdzielczosc zdiec do rozmiaru kalibracji
    # (stereo.left.image_size), wiec macierz Q pozostaje poprawna.
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img)

    disp = compute_disparity(left_rect, right_rect)

    # Etap 5: chmura punktow (budowana wprost z dysparycji + Q)
    xyz, colors = build_pointcloud(disp, stereo.Q, left_rect, max_depth_mm=5000.0)
    xyz, colors = filter_pointcloud(xyz, colors)

    # Etap 5-7: detekcja palety, segmentacja, pomiar
    pallet_result = detect_pallet(xyz)
    meas          = measure_object(xyz, pallet_result, noise_floor_mm=20.0)
    validation    = validate_measurement(meas)
    report_text   = generate_report(meas, validation)

    # Zapis plikow wynikowych
    save_ply(str(session.data_dir / "cloud.ply"), xyz, colors)
    (session.data_dir / "measurement_report.txt").write_text(report_text, encoding="utf-8")

    log.info("[%s] Pomiar OK: W=%.0f L=%.0f H=%.0f mm, valid=%s",
             session_id, meas.bbox.width, meas.bbox.length, meas.bbox.height,
             validation.passed)

    return MeasResult(
        width_mm=meas.bbox.width,
        length_mm=meas.bbox.length,
        height_mm=meas.bbox.height,
        volume_voxel_mm3=meas.volume.voxel_mm3,
        volume_bbox_mm3=meas.volume.bbox_mm3,
        volume_hull_mm3=meas.volume.hull_mm3,
        fill_ratio=meas.volume.fill_ratio,
        validation_passed=validation.passed,
        pallet_rms_mm=validation.pallet_plane_rms_mm,
        n_object_pts=meas.n_object_pts,
        n_pallet_inliers=meas.n_pallet_inliers,
        issues=list(validation.issues),
        report=report_text,
    )


async def measure_session(session_id: str) -> None:
    """Async wrapper - uruchamia pipeline pomiarowy i rozsyla wynik przez WS."""
    try:
        result = await asyncio.to_thread(_sync_measure, session_id)

        session = await store.get(session_id)
        session.meas_result = result
        await store.set_state(session_id, SessionState.DONE)

        await ws_manager.broadcast(session_id, {
            "event": "measurement_done",
            "width_mm": result.width_mm,
            "length_mm": result.length_mm,
            "height_mm": result.height_mm,
            "volume_voxel_l": result.volume_voxel_mm3 / 1e6,
            "validation_passed": result.validation_passed,
        })

    except Exception as exc:
        log.error("[%s] Blad pomiaru: %s", session_id, exc)
        await store.set_state(session_id, SessionState.READY)
        await ws_manager.broadcast(session_id, {
            "event": "error",
            "message": f"Pomiar nieudany: {exc}",
        })


# ===========================================================================
# POMIAR SYNTETYCZNY (bez zdiec, do testow API)
# ===========================================================================

def _sync_synthetic_measure() -> MeasResult:
    """Uruchamia pelny pipeline na danych syntetycznych (bez kamer)."""
    left_img, right_img, _, stereo = generate_synthetic_scene()

    img_size = (left_img.shape[1], left_img.shape[0])
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img, img_size)

    disp = compute_disparity(left_rect, right_rect)

    xyz, colors = build_pointcloud(disp, stereo.Q, left_rect, max_depth_mm=5000.0)
    xyz, colors = filter_pointcloud(xyz, colors)

    pallet_result = detect_pallet(xyz)
    meas          = measure_object(xyz, pallet_result, noise_floor_mm=20.0)
    validation    = validate_measurement(meas)
    report_text   = generate_report(meas, validation)

    return MeasResult(
        width_mm=meas.bbox.width,
        length_mm=meas.bbox.length,
        height_mm=meas.bbox.height,
        volume_voxel_mm3=meas.volume.voxel_mm3,
        volume_bbox_mm3=meas.volume.bbox_mm3,
        volume_hull_mm3=meas.volume.hull_mm3,
        fill_ratio=meas.volume.fill_ratio,
        validation_passed=validation.passed,
        pallet_rms_mm=validation.pallet_plane_rms_mm,
        n_object_pts=meas.n_object_pts,
        n_pallet_inliers=meas.n_pallet_inliers,
        issues=list(validation.issues),
        report=report_text,
    )


async def synthetic_measure() -> MeasResult:
    return await asyncio.to_thread(_sync_synthetic_measure)
