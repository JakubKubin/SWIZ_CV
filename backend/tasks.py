# backend/tasks.py
"""Zadania wykonywane w tle (kalibracja, pomiar) + zarzadzanie WebSocket.

Zadania CPU-intensive sa uruchamiane w puli watkow przez asyncio.to_thread(),
zeby nie blokowac petli zdarzen uvicorn.

WebSocketManager zarzadza polaczeniami all devices w ramach sesji.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

import config
from calibration import calibrate_stereo, save_params, load_params, baseline_warning
from disparity import (
    rectify_pair, compute_disparity, disparity_to_depth,
    colormap_disparity, colormap_depth, draw_epipolar_check,
)
from pointcloud import (
    build_pointcloud, filter_pointcloud, save_ply,
    render_topdown, render_sideview,
)
from pallet import detect_pallet
from measurement import measure_object, validate_measurement, generate_report
from pipeline import generate_synthetic_scene

from .session import (
    Session, SessionState, CalibResult, MeasResult, store,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

log = logging.getLogger(__name__)


# ===========================================================================
# WebSocketManager
# ===========================================================================

class WSManager:
    """Utrzymuje aktywne polaczenia WebSocket per (session_id, device_id)."""

    def __init__(self) -> None:
        self._conns: dict[tuple[str, str], WebSocket] = {}

    async def connect(self, ws: WebSocket, session_id: str, device_id: str) -> None:
        key = (session_id, device_id)
        old = self._conns.get(key)
        if old is not None:
            # Close the stale socket so the OS can reclaim the file descriptor.
            try:
                await old.close(code=1001)
            except Exception:
                pass
        await ws.accept()
        self._conns[key] = ws

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
        targets: list[tuple[str, WebSocket]] = [
            (did, ws) for (sid, did), ws in self._conns.items() if sid == session_id
        ]
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
# Pomocniki plikowe
# ===========================================================================

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _sorted_frames(directory: Path, prefix: str) -> list[Path]:
    """Glob all images with *prefix*_NNNN.* and sort by numeric index.

    Sorting purely by the numeric part of the stem avoids the broken
    "sorted-by-ext then concatenated" pattern: mixed extensions within
    the same directory are handled correctly regardless of their order.
    """
    frames = [p for p in directory.glob(f"{prefix}_*.*")
              if p.suffix.lower() in _IMG_SUFFIXES]
    return sorted(frames, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def _latest_image(directory: Path, prefix: str) -> Path:
    """Return the most-recently-modified image matching *prefix*_*.*."""
    imgs = [p for p in directory.glob(f"{prefix}_*.*")
            if p.suffix.lower() in _IMG_SUFFIXES]
    if not imgs:
        raise FileNotFoundError(f"Brak zdiec ({prefix}_*) w: {directory}")
    return max(imgs, key=lambda p: p.stat().st_mtime)


# ===========================================================================
# KALIBRACJA
# ===========================================================================

def _sync_calibrate(session_id: str) -> CalibResult:
    """Synchroniczna kalibracja stereo - uruchamiana w watku roboczym.

    Wczytuje obrazy kalibracyjne z dysku, uruchamia calibrate_stereo()
    i zapisuje wynik do stereo.json.
    """
    session = store.get_sync(session_id)

    left = session.left_camera()
    right = session.right_camera()
    if left is None or right is None:
        raise ValueError("Potrzeba 2 urządzeń z kamerą do kalibracji stereo")

    left_paths  = _sorted_frames(session.calib_dir(left.device_id),  "frame")
    right_paths = _sorted_frames(session.calib_dir(right.device_id), "frame")

    pairs = list(zip(left_paths, right_paths))
    if len(pairs) < 3:
        raise ValueError(
            f"Za malo par kalibracyjnych: {len(pairs)} < 3. "
            f"Lewa: {len(left_paths)} kl., Prawa: {len(right_paths)} kl."
        )

    log.info("Kalibracja: %d par klatek (%s / %s)", len(pairs),
             left.device_id, right.device_id)

    debug_dir = session.data_dir / "debug" / "corners"
    stereo = calibrate_stereo(
        [str(p) for p, _ in pairs],
        [str(p) for _, p in pairs],
        debug_dir=debug_dir,
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
        # Persist calib_result first; set_state will also call _write_meta,
        # but being explicit makes the ordering obvious and future-proof.
        await store.save(session_id)
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
      2. Rektyfikacja → dysparycja SGBM → mapa glebokosci → chmura punktow
      3. Detekcja palety (RANSAC) → segmentacja → pomiar wymiarow
    Wszystkie wyniki posrednie sa zapisywane do session.data_dir.
    """
    session = store.get_sync(session_id)
    out = session.data_dir

    if session.calib_result is None:
        raise ValueError("Brak kalibracji - najpierw wykonaj kalibracje")

    left = session.left_camera()
    right = session.right_camera()
    if left is None or right is None:
        raise ValueError("Potrzeba 2 urządzeń z kamerą do pomiaru")

    stereo = load_params(session.calib_result.params_path, stereo=True)

    left_path  = _latest_image(session.capture_dir(left.device_id),  "capture")
    right_path = _latest_image(session.capture_dir(right.device_id), "capture")

    log.info("[%s] Pomiar: left=%s right=%s", session_id, left_path.name, right_path.name)

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None or right_img is None:
        raise IOError(f"Nie mozna wczytac zdiec: {left_path}, {right_path}")

    # Etap 1: zachowaj kopie obrazow wejsciowych
    cv2.imwrite(str(out / "input_left.png"), left_img)
    cv2.imwrite(str(out / "input_right.png"), right_img)

    # Etap 2: rektyfikacja
    # rectify_pair sam dopasowuje rozdzielczosc do rozmiaru kalibracji
    # (stereo.left.image_size), wiec macierz Q pozostaje poprawna.
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img)
    cv2.imwrite(str(out / "left_rect.png"), left_rect)
    cv2.imwrite(str(out / "right_rect.png"), right_rect)
    cv2.imwrite(str(out / "epipolar_check.png"),
                draw_epipolar_check(left_rect, right_rect))

    # Etap 3: mapa dysparycji SGBM
    disp = compute_disparity(left_rect, right_rect)
    np.save(str(out / "disparity.npy"), disp)
    cv2.imwrite(str(out / "disparity_color.png"), colormap_disparity(disp))

    # Etap 4: mapa glebokosci przez macierz Q
    depth = disparity_to_depth(disp, stereo.Q, max_depth_mm=config.MAX_DEPTH_MM)
    np.save(str(out / "depth_mm.npy"), depth)
    cv2.imwrite(str(out / "depth_color.png"), colormap_depth(depth))

    # Etap 5: chmura punktow (budowana wprost z dysparycji + Q)
    xyz_raw, colors_raw = build_pointcloud(disp, stereo.Q, left_rect,
                                           max_depth_mm=config.MAX_DEPTH_MM)
    xyz, colors = filter_pointcloud(xyz_raw, colors_raw)
    save_ply(str(out / "cloud.ply"), xyz, colors)
    cv2.imwrite(str(out / "view_topdown.png"), render_topdown(xyz, colors))
    cv2.imwrite(str(out / "view_sideview.png"), render_sideview(xyz, colors))

    # Etap 6: detekcja palety RANSAC + SVD
    pallet_result = detect_pallet(xyz)
    pallet_data = {
        "plane_normal": pallet_result.plane.normal.tolist(),
        "plane_d": float(pallet_result.plane.d),
        "rms_residual_mm": float(pallet_result.plane.rms_residual),
        "n_inliers": int(pallet_result.plane.inlier_mask.sum()),
        "n_roi_pts": int(pallet_result.roi_mask.sum()),
        "centroid_mm": pallet_result.centroid.tolist(),
        "R_cam_to_pallet": pallet_result.R.tolist(),
    }
    (out / "pallet.json").write_text(json.dumps(pallet_data, indent=2), encoding="utf-8")

    # Etap 7+8: segmentacja, pomiar, walidacja
    meas       = measure_object(xyz, pallet_result, noise_floor_mm=config.NOISE_FLOOR_MM)
    validation = validate_measurement(meas)
    report_text = generate_report(meas, validation)
    (out / "measurement_report.txt").write_text(report_text, encoding="utf-8")

    # Zapis zbiorczego JSON z metrykami kazdego etapu
    pipeline_steps = {
        "input": {
            "left_file": left_path.name,
            "right_file": right_path.name,
            "calibration_reproj_error_px": float(stereo.reproj_error),
        },
        "rectification": {
            "image_size": list(left_rect.shape[:2][::-1]),
        },
        "disparity": {
            "valid_px": int((disp > 0).sum()),
            "total_px": int(disp.size),
            "coverage_pct": round(100.0 * float((disp > 0).sum()) / disp.size, 1),
        },
        "depth": {
            "valid_px": int((depth > 0).sum()),
            "median_mm": round(float(np.median(depth[depth > 0])), 1) if (depth > 0).any() else None,
        },
        "pointcloud": {
            "n_pts_raw": int(len(xyz_raw)),
            "n_pts_filtered": int(len(xyz)),
        },
        "pallet": {
            "rms_residual_mm": float(pallet_result.plane.rms_residual),
            "n_inliers": int(pallet_result.plane.inlier_mask.sum()),
            "n_roi_pts": int(pallet_result.roi_mask.sum()),
        },
        "measurement": {
            "n_object_pts": int(meas.n_object_pts),
            "width_mm": round(float(meas.bbox.width), 1),
            "length_mm": round(float(meas.bbox.length), 1),
            "height_mm": round(float(meas.bbox.height), 1),
            "volume_voxel_l": round(float(meas.volume.voxel_mm3 / 1e6), 4),
            "volume_bbox_l": round(float(meas.volume.bbox_mm3 / 1e6), 4),
        },
        "validation": {
            "passed": bool(validation.passed),
            "issues": list(validation.issues),
        },
    }
    (out / "pipeline_steps.json").write_text(
        json.dumps(pipeline_steps, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
        await store.save(session_id)
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

    xyz, colors = build_pointcloud(disp, stereo.Q, left_rect,
                                   max_depth_mm=config.MAX_DEPTH_MM)
    xyz, colors = filter_pointcloud(xyz, colors)

    pallet_result = detect_pallet(xyz)
    meas          = measure_object(xyz, pallet_result,
                                   noise_floor_mm=config.NOISE_FLOOR_MM)
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
