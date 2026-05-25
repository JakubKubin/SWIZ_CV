# backend/main.py
"""FastAPI backend dla systemu stereowizyjnego pomiaru na europalecie.

Uruchomienie:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Dokumentacja interaktywna (Swagger UI):
    http://localhost:8000/docs

API:
  POST   /sessions                              - utwórz sesję
  POST   /sessions/{sid}/join                   - dołącz urządzenie
  GET    /sessions/{sid}                        - status sesji
  DELETE /sessions/{sid}                        - usuń sesję
  GET    /sessions                              - lista sesji

  POST   /sessions/{sid}/calibration/images     - prześlij obraz kalibracyjny
  POST   /sessions/{sid}/calibration/compute    - uruchom kalibrację
  GET    /sessions/{sid}/calibration            - status kalibracji

  POST   /sessions/{sid}/capture/trigger        - roześlij komendę przechwycenia
  POST   /sessions/{sid}/capture/images         - prześlij obraz pomiarowy

  POST   /sessions/{sid}/measure                - uruchom pipeline pomiaru
  GET    /sessions/{sid}/measurement            - wyniki pomiaru
  GET    /sessions/{sid}/measurement/report     - raport tekstowy

  POST   /measure/synthetic                     - test bez kamer (dane syntetyczne)
  GET    /health                                - health check

  WS     /ws/{sid}/{device_id}                  - synchronizacja w czasie rzeczywistym
"""

import asyncio
import io
import logging
import time

import cv2 as _cv2
from PIL import Image as _PilImage, ImageOps as _PilImageOps
from calibration import find_corners as _find_corners
import config

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, Query,
    WebSocket, WebSocketDisconnect, Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from .schemas import (
    JoinRequest, SessionOut, DeviceOut,
    CalibStatusOut, TriggerRequest, TriggerOut,
    MeasurementOut, HealthOut, DevicePatchRequest, CameraAssignRequest,
)
from .session import store, SessionState
from .tasks import ws_manager, calibrate_session, measure_session, synthetic_measure

from logging_setup import setup_logging

setup_logging()
log = logging.getLogger(__name__)


# ===========================================================================
# Aplikacja FastAPI
# ===========================================================================

app = FastAPI(
    title="Stereo Vision API",
    description="Backend systemu stereowizyjnego pomiaru obiektów na europalecie",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # w produkcji podaj konkretne originy
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Pomocnicy
# ===========================================================================

async def _get_or_404(session_id: str):
    """Pobiera sesję lub zwraca HTTP 404."""
    try:
        return await store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Sesja '{session_id}' nie istnieje")


def _normalize_image(content: bytes) -> bytes:
    """Synchroniczna konwersja: EXIF-rotate + zapis PNG. Wywoływana w thread pool.

    Obsługuje również automatyczny obrót całego obrazu na podstawie config.IMAGE_ROTATE.
    """
    img = _PilImageOps.exif_transpose(_PilImage.open(io.BytesIO(content))).convert("RGB")

    rot_angle = getattr(config, "IMAGE_ROTATE", 0)
    if rot_angle == 90:
        img = img.transpose(_PilImage.ROTATE_270)  # 90 CW (Pillow rotation is CCW by default, so transposing 270 CCW rotates it 90 CW)
    elif rot_angle == 180:
        img = img.transpose(_PilImage.ROTATE_180)
    elif rot_angle == 270:
        img = img.transpose(_PilImage.ROTATE_90)   # 90 CCW

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _detect_corners_in_file(path) -> bool:
    """Returns True if checkerboard corners are found. Called in a thread pool."""
    img = _cv2.imread(str(path))
    if img is None:
        return False
    try:
        return _find_corners(img) is not None
    except Exception:
        return False


def _session_to_out(session) -> SessionOut:
    """Konwertuje obiekt Session na Pydantic SessionOut."""
    left = session.left_camera()
    right = session.right_camera()
    return SessionOut(
        session_id=session.session_id,
        state=session.state,
        devices=[
            DeviceOut(
                device_id=d.device_id,
                mac=d.mac,
                is_leader=d.is_leader,
                is_camera=d.is_camera,
                joined_at=d.joined_at,
                ws_connected=d.ws_connected,
                calib_frame_count=d.calib_frame_count,
                capture_frame_count=d.capture_frame_count,
            )
            for d in session.devices.values()
        ],
        created_at=session.created_at,
        has_calibration=session.calib_result is not None,
        has_measurement=session.meas_result is not None,
        left_device_id=left.device_id if left else None,
        right_device_id=right.device_id if right else None,
    )


def _meas_to_out(result) -> MeasurementOut:
    return MeasurementOut(
        validation_passed=result.validation_passed,
        width_mm=result.width_mm,
        length_mm=result.length_mm,
        height_mm=result.height_mm,
        volume_voxel_mm3=result.volume_voxel_mm3,
        volume_bbox_mm3=result.volume_bbox_mm3,
        volume_hull_mm3=result.volume_hull_mm3,
        fill_ratio=result.fill_ratio,
        pallet_rms_mm=result.pallet_rms_mm,
        n_object_pts=result.n_object_pts,
        n_pallet_inliers=result.n_pallet_inliers,
        issues=result.issues,
        report=result.report,
    )


# ===========================================================================
# HEALTH
# ===========================================================================

@app.get("/health", response_model=HealthOut, tags=["Utility"])
async def health():
    """Sprawdza czy serwer dziala."""
    return HealthOut()


# ===========================================================================
# SESJE
# ===========================================================================

@app.post("/sessions", response_model=SessionOut, status_code=201, tags=["Sesje"])
async def create_session():
    """Tworzy nową sesję pomiarową. Zwraca session_id potrzebny do dalszych wywolan."""
    session = await store.create()
    log.info("Nowa sesja: %s", session.session_id)
    return _session_to_out(session)


@app.get("/sessions", response_model=list[SessionOut], tags=["Sesje"])
async def list_sessions():
    """Zwraca liste wszystkich aktywnych sesji (do debugowania)."""
    sessions = await store.list_all()
    return [_session_to_out(s) for s in sessions]


@app.get("/sessions/{session_id}", response_model=SessionOut, tags=["Sesje"])
async def get_session(session_id: str):
    """Zwraca aktualny stan sesji (urzadzenia, etap, wyniki)."""
    session = await _get_or_404(session_id)
    return _session_to_out(session)


@app.post("/sessions/{session_id}/join", response_model=SessionOut, tags=["Sesje"])
async def join_session(session_id: str, body: JoinRequest):
    """Rejestruje urzadzenie w sesji.

    - Pierwsza kamera (lewa): `is_leader=true`
    - Druga kamera (prawa): `is_leader=false`
    - Maksymalnie 2 urzadzenia na sesje.
    """
    session = await _get_or_404(session_id)

    if body.device_id in session.devices:
        return _session_to_out(session)

    if session.is_full():
        raise HTTPException(status_code=409, detail="Sesja jest pełna (maks. 10 urządzeń)")

    # Tylko jedno urzadzenie moze byc leaderem.
    # Wyjątek 1: rozłączony lider może być zastąpiony przez nowe urządzenie.
    # Wyjątek 2: urządzenie bez kamery może przejąć rolę lidera od urządzenia z kamerą.
    existing_leader = session.leader()
    if body.is_leader and existing_leader is not None:
        can_takeover = (
            not existing_leader.ws_connected
            or (not body.is_camera and existing_leader.is_camera)
        )
        if can_takeover:
            existing_leader.is_leader = False
            log.info("[%s] Lider przejęty przez %s (poprzedni: %s, ws=%s)", session_id, body.device_id, existing_leader.device_id, existing_leader.ws_connected)
        else:
            raise HTTPException(status_code=409, detail="Leader już istnieje w tej sesji")

    # Jeśli sesja nie ma lidera, pierwszy dołączający przejmuje rolę automatycznie
    if session.leader() is None:
        body = body.model_copy(update={"is_leader": True})
        log.info("[%s] Brak lidera - %s automatycznie zostaje liderem", session_id, body.device_id)

    from .session import Device
    device = Device(
        device_id=body.device_id,
        mac=body.mac,
        is_leader=body.is_leader,
        is_camera=body.is_camera,
    )
    session.devices[body.device_id] = device
    if body.is_camera:
        session.calib_dir(body.device_id).mkdir(parents=True, exist_ok=True)
        session.capture_dir(body.device_id).mkdir(parents=True, exist_ok=True)
    await store.save(session_id)

    log.info("[%s] Dołączyło urządzenie: %s (leader=%s)", session_id, body.device_id, body.is_leader)

    await ws_manager.broadcast(session_id, {
        "event": "device_joined",
        "device_id": body.device_id,
        "is_leader": body.is_leader,
        "total_devices": len(session.devices),
    })

    return _session_to_out(session)


@app.delete("/sessions/{session_id}", status_code=204, tags=["Sesje"])
async def delete_session(session_id: str):
    """Usuwa sesję i wszystkie jej dane (obrazy, pliki wynikowe)."""
    deleted = await store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Sesja '{session_id}' nie istnieje")
    log.info("Sesja %s usunięta", session_id)
    return Response(status_code=204)


@app.delete("/sessions/{session_id}/devices/{device_id}", status_code=204, tags=["Sesje"])
async def leave_session(
    session_id: str,
    device_id: str,
    requester_id: str | None = Query(default=None, description="ID urządzenia żądającego usunięcia; lider może usunąć dowolne urządzenie"),
):
    """Wypisuje jedno urządzenie z sesji.

    Bez requester_id: urządzenie samo się wypisuje.
    Z requester_id: wymagane aby requester był liderem sesji — może usunąć dowolne urządzenie.
    """
    session = await _get_or_404(session_id)

    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    if requester_id is not None and requester_id != device_id:
        requester = session.devices.get(requester_id)
        if requester is None:
            raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
        if not requester.is_leader:
            raise HTTPException(status_code=403, detail="Tylko lider może usuwać inne urządzenia")

    del session.devices[device_id]
    await store.save(session_id)
    log.info("[%s] Urządzenie usunięte: %s (przez: %s, pozostało: %d)",
             session_id, device_id, requester_id or device_id, len(session.devices))

    await ws_manager.broadcast(session_id, {
        "event": "device_left",
        "device_id": device_id,
        "remaining": len(session.devices),
    })

    return Response(status_code=204)


@app.post("/sessions/{session_id}/devices/{device_id}/promote", response_model=SessionOut, tags=["Sesje"])
async def promote_device(
    session_id: str,
    device_id: str,
    requester_id: str = Query(..., description="ID bieżącego lidera autoryzującego operację"),
):
    """Przekazuje rolę lidera innemu urządzeniu w sesji.

    Tylko aktualny lider może wywołać ten endpoint.
    """
    session = await _get_or_404(session_id)

    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może przekazać rolę lidera")

    target = session.devices.get(device_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")
    if target.is_leader:
        return _session_to_out(session)

    requester.is_leader = False
    target.is_leader = True
    await store.save(session_id)

    log.info("[%s] Lider zmieniony: %s → %s", session_id, requester_id, device_id)
    await ws_manager.broadcast(session_id, {
        "event": "leader_changed",
        "old_leader": requester_id,
        "new_leader": device_id,
    })

    return _session_to_out(session)


@app.patch("/sessions/{session_id}/devices/{device_id}", response_model=SessionOut, tags=["Sesje"])
async def patch_device(
    session_id: str,
    device_id: str,
    body: DevicePatchRequest,
    requester_id: str = Query(..., description="ID lidera autoryzującego operację"),
):
    """Zmienia właściwości urządzenia (is_camera) po dołączeniu do sesji.

    Tylko lider może zmieniać typ urządzenia.
    Przy zmianie na is_camera=True tworzone są katalogi kalibracji i przechwytywania.
    """
    session = await _get_or_404(session_id)

    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może zmieniać typ urządzenia")

    target = session.devices.get(device_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    target.is_camera = body.is_camera
    if body.is_camera:
        session.calib_dir(device_id).mkdir(parents=True, exist_ok=True)
        session.capture_dir(device_id).mkdir(parents=True, exist_ok=True)
    await store.save(session_id)

    log.info("[%s] Urządzenie %s: is_camera=%s (przez: %s)", session_id, device_id, body.is_camera, requester_id)
    await ws_manager.broadcast(session_id, {
        "event": "device_updated",
        "device_id": device_id,
        "is_camera": body.is_camera,
    })

    return _session_to_out(session)


@app.put("/sessions/{session_id}/cameras", response_model=SessionOut, tags=["Sesje"])
async def assign_cameras(
    session_id: str,
    body: CameraAssignRequest,
    requester_id: str = Query(..., description="ID lidera autoryzującego operację"),
):
    """Ręcznie przypisuje urządzenia do roli lewej i prawej kamery stereo.

    Tylko lider może wywołać ten endpoint. Przypisanie jest zapisywane w sesji
    i nadpisuje domyślny podział (lider=lewa, follower=prawa).
    Wymagana jest ponowna kalibracja po zmianie.
    """
    session = await _get_or_404(session_id)

    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może przypisywać kamery")

    for did, role in [(body.left_device_id, "lewa"), (body.right_device_id, "prawa")]:
        dev = session.devices.get(did)
        if dev is None:
            raise HTTPException(status_code=404, detail=f"Urządzenie '{did}' nie jest w sesji")
        if not dev.is_camera:
            raise HTTPException(status_code=422, detail=f"Urządzenie '{did}' nie jest kamerą ({role})")

    if body.left_device_id == body.right_device_id:
        raise HTTPException(status_code=422, detail="Lewa i prawa kamera muszą być różnymi urządzeniami")

    session.left_device_id = body.left_device_id
    session.right_device_id = body.right_device_id
    await store.save(session_id)

    log.info("[%s] Kamery przypisane: lewa=%s prawa=%s (przez: %s)",
             session_id, body.left_device_id, body.right_device_id, requester_id)
    await ws_manager.broadcast(session_id, {
        "event": "cameras_assigned",
        "left_device_id": body.left_device_id,
        "right_device_id": body.right_device_id,
    })

    return _session_to_out(session)


# ===========================================================================
# KALIBRACJA
# ===========================================================================

@app.post(
    "/sessions/{session_id}/calibration/images",
    status_code=201,
    tags=["Kalibracja"],
)
async def upload_calib_image(
    session_id: str,
    device_id: str = Form(..., description="Identyfikator urządzenia przesyłającego obraz"),
    file: UploadFile = File(..., description="Obraz szachownicy kalibracyjnej (JPG lub PNG)"),
):
    """Przesyła jeden obraz kalibracyjny (szachownica) dla danego urządzenia.

    Obrazy sa numerowane sekwencyjnie: frame_0000.png, frame_0001.png, ...
    Wywołaj co najmniej 3 razy dla każdego urządzenia przed uruchomieniem kalibracji.
    """
    session = await _get_or_404(session_id)

    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    device = session.devices[device_id]
    calib_dir = session.calib_dir(device_id)
    calib_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()

    # PIL is CPU-bound — run in thread pool so the event loop stays free
    try:
        png_bytes = await asyncio.to_thread(_normalize_image, content)
    except Exception:
        log.warning("[%s] Nie można przetworzyć obrazu kalibracyjnego %s: %s", session_id, device_id, "błąd konwersji do PNG - zapis oryginału")
        png_bytes = content

    async with session._lock:
        save_path = calib_dir / f"frame_{device.calib_frame_count:04d}.png"
        await asyncio.to_thread(save_path.write_bytes, png_bytes)
        device.calib_frame_count += 1
        frame_index = device.calib_frame_count - 1
        total = device.calib_frame_count

    # Detect corners outside the lock — CPU-intensive, reads the just-saved file.
    detected = await asyncio.to_thread(_detect_corners_in_file, save_path)
    session.calib_detection.setdefault(device_id, {})[frame_index] = detected

    await store.save(session_id)
    log.info("[%s] Kalibracja %s: klatka %d zapisana (%d B, wykryto=%s)",
             session_id, device_id, frame_index, len(png_bytes), detected)

    await ws_manager.broadcast(session_id, {
        "event": "calib_frame_uploaded",
        "device_id": device_id,
        "total_frames": total,
    })

    return {"device_id": device_id, "frame_index": frame_index, "total_frames": total}


@app.delete(
    "/sessions/{session_id}/calibration/images/{target_device_id}",
    status_code=200,
    tags=["Kalibracja"],
)
async def delete_calib_images(
    session_id: str,
    target_device_id: str,
    requester_id: str = Query(..., description="ID lidera autoryzującego operację"),
):
    """Usuwa wszystkie zdjęcia kalibracyjne dla danego urządzenia i resetuje licznik.

    Tylko lider sesji może wywołać ten endpoint.
    """
    session = await _get_or_404(session_id)

    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może usuwać zdjęcia kalibracyjne")

    if target_device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{target_device_id}' nie jest w sesji")

    calib_dir = session.calib_dir(target_device_id)
    deleted = 0
    if calib_dir.exists():
        for f in calib_dir.glob("frame_*"):
            f.unlink()
            deleted += 1

    session.devices[target_device_id].calib_frame_count = 0
    await store.save(session_id)

    log.info("[%s] Usunięto %d zdjęć kalibracyjnych urządzenia %s (przez: %s)",
             session_id, deleted, target_device_id, requester_id)

    await ws_manager.broadcast(session_id, {
        "event": "calib_images_cleared",
        "device_id": target_device_id,
        "deleted_count": deleted,
    })

    return {"device_id": target_device_id, "deleted_count": deleted}


@app.get(
    "/sessions/{session_id}/calibration/images/{device_id}",
    tags=["Kalibracja"],
)
async def list_calib_images(session_id: str, device_id: str):
    """Zwraca posortowaną listę indeksów zdjęć kalibracyjnych dla urządzenia."""
    session = await _get_or_404(session_id)
    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")
    calib_dir = session.calib_dir(device_id)
    frames = sorted(calib_dir.glob("frame_*.png")) if calib_dir.exists() else []
    return [{"index": int(f.stem.split("_")[1]), "filename": f.name} for f in frames]


@app.get(
    "/sessions/{session_id}/calibration/images/{device_id}/{frame_index}",
    tags=["Kalibracja"],
)
async def get_calib_image(session_id: str, device_id: str, frame_index: int):
    """Serwuje pojedyncze zdjęcie kalibracyjne."""
    session = await _get_or_404(session_id)
    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")
    path = session.calib_dir(device_id) / f"frame_{frame_index:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Klatka {frame_index} nie istnieje")
    return FileResponse(path, media_type="image/png")


@app.get(
    "/sessions/{session_id}/calibration/detection",
    tags=["Kalibracja"],
)
async def get_calib_detection(session_id: str):
    """Zwraca zakeszowane wyniki wykrycia szachownicy dla każdej klatki.

    Wyniki są zapisywane podczas uploadu — bez ponownego uruchamiania OpenCV.
    Zwraca słownik: device_id → {frame_index: bool}.
    """
    session = await _get_or_404(session_id)
    return session.calib_detection


@app.delete(
    "/sessions/{session_id}/calibration/pairs/{frame_index}",
    status_code=200,
    tags=["Kalibracja"],
)
async def delete_calib_pair(
    session_id: str,
    frame_index: int,
    requester_id: str = Query(..., description="ID lidera autoryzującego operację"),
):
    """Usuwa parę klatek kalibracyjnych (ten sam indeks ze wszystkich urządzeń-kamer)."""
    session = await _get_or_404(session_id)
    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może usuwać klatki kalibracyjne")

    camera_devices = [d for d in session.devices.values() if d.is_camera]
    deleted_from = []
    counts = {}
    for dev in camera_devices:
        path = session.calib_dir(dev.device_id) / f"frame_{frame_index:04d}.png"
        if path.exists():
            path.unlink()
            deleted_from.append(dev.device_id)
        remaining = list(session.calib_dir(dev.device_id).glob("frame_*.png"))
        dev.calib_frame_count = len(remaining)
        counts[dev.device_id] = dev.calib_frame_count
        session.calib_detection.get(dev.device_id, {}).pop(frame_index, None)

    await store.save(session_id)
    log.info("[%s] Usunięto parę kalibracyjną %d (przez: %s, urządzenia: %s)",
             session_id, frame_index, requester_id, deleted_from)

    await ws_manager.broadcast(session_id, {
        "event": "calib_pair_cleared",
        "frame_index": frame_index,
        "deleted_from": deleted_from,
        "counts": counts,
    })
    return {"frame_index": frame_index, "deleted_from": deleted_from, "counts": counts}


@app.post(
    "/sessions/{session_id}/calibration/compute",
    status_code=202,
    tags=["Kalibracja"],
)
async def compute_calibration(session_id: str):
    """Uruchamia obliczenie kalibracji stereo (metoda Zhanga) w tle.

    Wynik dostepny przez GET /sessions/{sid}/calibration i przez WebSocket
    (event: "calibration_done" lub "error").

    Wymagania:
    - 2 urządzenia w sesji
    - co najmniej 3 sparowane klatki kalibracyjne per urządzenie
    """
    session = await _get_or_404(session_id)

    camera_devices = [d for d in session.devices.values() if d.is_camera]
    if len(camera_devices) < 2:
        raise HTTPException(
            status_code=409,
            detail="Potrzeba 2 urządzeń z kamerą do kalibracji stereo",
        )

    if session.state == SessionState.CALIBRATING:
        raise HTTPException(status_code=409, detail="Kalibracja już trwa")

    if session.state == SessionState.PROCESSING:
        raise HTTPException(status_code=409, detail="Pipeline pomiarowy trwa - poczekaj")

    min_frames = session.min_calib_frames()
    if min_frames < 3:
        raise HTTPException(
            status_code=409,
            detail=f"Za mało klatek kalibracyjnych (min. 3 na urządzenie, teraz: {min_frames})",
        )

    await store.set_state(session_id, SessionState.CALIBRATING)
    asyncio.create_task(calibrate_session(session_id))

    log.info("[%s] Kalibracja uruchomiona w tle", session_id)
    return {"message": "Kalibracja uruchomiona", "state": SessionState.CALIBRATING}


@app.post(
    "/sessions/{session_id}/calibration/trigger",
    response_model=TriggerOut,
    tags=["Kalibracja"],
)
async def trigger_calib_capture(session_id: str, body: TriggerRequest):
    """Broadcast zsynchronizowanego triggera kalibracyjnego do wszystkich urządzeń."""
    session = await _get_or_404(session_id)
    if session.state == SessionState.CALIBRATING:
        raise HTTPException(status_code=409, detail="Kalibracja w toku — poczekaj")
    capture_at = time.time() + body.delay_ms / 1000.0
    await ws_manager.broadcast(session_id, {
        "event": "calib_trigger",
        "at": capture_at,
        "delay_ms": body.delay_ms,
    })
    log.info("[%s] Trigger kalibracji: at=%.3f (delay=%d ms)", session_id, capture_at, body.delay_ms)
    return TriggerOut(at=capture_at, delay_ms=body.delay_ms)


@app.get("/sessions/{session_id}/calibration", response_model=CalibStatusOut, tags=["Kalibracja"])
async def get_calibration_status(session_id: str):
    """Zwraca status kalibracji: stan, błąd reprojekcji, komunikat."""
    session = await _get_or_404(session_id)

    if session.state == SessionState.CALIBRATING:
        message = "Kalibracja w toku - poczekaj na wynik WebSocket"
    elif session.calib_result:
        message = f"Kalibracja OK (reproj_error={session.calib_result.reproj_error:.3f} px)"
    else:
        message = "Brak kalibracji - prześlij obrazy i wywołaj /compute"

    return CalibStatusOut(
        state=session.state,
        reproj_error=session.calib_result.reproj_error if session.calib_result else None,
        message=message,
    )


# ===========================================================================
# PRZECHWYTYWANIE
# ===========================================================================

@app.post(
    "/sessions/{session_id}/capture/trigger",
    response_model=TriggerOut,
    tags=["Przechwytywanie"],
)
async def trigger_capture(session_id: str, body: TriggerRequest):
    """Rozsyła do wszystkich urzadzeń komendę jednoczesnego przechwycenia.

    Urządzenia powinny wykonac zdjęcie dokładnie o podanym timestamps `at`
    (Unix timestamp w sekundach). Synchronizacja czasu NTP zapewnia spójnosc.

    WebSocket event: `{"event": "capture_trigger", "at": <timestamp>}`
    """
    session = await _get_or_404(session_id)

    if session.state not in (SessionState.READY, SessionState.DONE):
        raise HTTPException(
            status_code=409,
            detail=f"Sesja nie jest gotowa do przechwycenia (stan: {session.state})",
        )

    capture_at = time.time() + body.delay_ms / 1000.0

    await ws_manager.broadcast(session_id, {
        "event": "capture_trigger",
        "at": capture_at,
        "delay_ms": body.delay_ms,
    })

    log.info("[%s] Trigger przechwycenia: at=%.3f (delay=%d ms)",
             session_id, capture_at, body.delay_ms)

    return TriggerOut(at=capture_at, delay_ms=body.delay_ms)


@app.post(
    "/sessions/{session_id}/capture/images",
    status_code=201,
    tags=["Przechwytywanie"],
)
async def upload_capture_image(
    session_id: str,
    device_id: str = Form(..., description="Identyfikator urządzenia"),
    file: UploadFile = File(..., description="Zdjęcie pomiarowe (JPG lub PNG)"),
):
    """Przesyła zdjęcie pomiarowe (obiektu na palecie) z danego urządzenia.

    Obrazy są numerowane sekwencyjnie: capture_0000.png, capture_0001.png, ...
    Pipeline pomiarowy zawsze używa najnowszego zdjecia dla każdego urządzenia.
    """
    session = await _get_or_404(session_id)

    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    if session.state not in (SessionState.READY, SessionState.DONE):
        raise HTTPException(
            status_code=409,
            detail=f"Sesja nie jest w stanie READY/DONE (teraz: {session.state})",
        )

    device = session.devices[device_id]
    capture_dir = session.capture_dir(device_id)
    capture_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()

    try:
        png_bytes = await asyncio.to_thread(_normalize_image, content)
    except Exception:
        log.warning("[%s] Nie można przetworzyć obrazu przechwycenia %s: %s",
                    session_id, device_id, "błąd konwersji do PNG - zapis oryginału")
        png_bytes = content

    async with session._lock:
        save_path = capture_dir / f"capture_{device.capture_frame_count:04d}.png"
        await asyncio.to_thread(save_path.write_bytes, png_bytes)
        device.capture_frame_count += 1
        frame_index = device.capture_frame_count - 1
        total = device.capture_frame_count

    await store.save(session_id)
    log.info("[%s] Zdjecie %s: capture %d (%d B)",
             session_id, device_id, frame_index, len(png_bytes))

    return {"device_id": device_id, "frame_index": frame_index, "total_frames": total}


@app.delete(
    "/sessions/{session_id}/capture/images/{target_device_id}",
    status_code=200,
    tags=["Przechwytywanie"],
)
async def delete_capture_images(
    session_id: str,
    target_device_id: str,
    requester_id: str = Query(..., description="ID lidera autoryzującego operację"),
):
    """Usuwa wszystkie zdjęcia pomiarowe dla danego urządzenia i resetuje licznik.

    Tylko lider sesji może wywołać ten endpoint.
    """
    session = await _get_or_404(session_id)

    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może usuwać zdjęcia pomiarowe")

    if target_device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{target_device_id}' nie jest w sesji")

    capture_dir = session.capture_dir(target_device_id)
    deleted = 0
    if capture_dir.exists():
        for f in capture_dir.glob("capture_*"):
            f.unlink()
            deleted += 1

    session.devices[target_device_id].capture_frame_count = 0
    await store.save(session_id)

    log.info("[%s] Usunięto %d zdjęć pomiarowych urządzenia %s (przez: %s)",
             session_id, deleted, target_device_id, requester_id)

    await ws_manager.broadcast(session_id, {
        "event": "capture_images_cleared",
        "device_id": target_device_id,
        "deleted_count": deleted,
    })

    return {"device_id": target_device_id, "deleted_count": deleted}


@app.get(
    "/sessions/{session_id}/capture/images/{device_id}",
    tags=["Przechwytywanie"],
)
async def list_capture_images(session_id: str, device_id: str):
    """Zwraca posortowaną listę indeksów zdjęć pomiarowych dla urządzenia."""
    session = await _get_or_404(session_id)
    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")
    capture_dir = session.capture_dir(device_id)
    frames = sorted(capture_dir.glob("capture_*.png")) if capture_dir.exists() else []
    return [{"index": int(f.stem.split("_")[1]), "filename": f.name} for f in frames]


@app.get(
    "/sessions/{session_id}/capture/images/{device_id}/{frame_index}",
    tags=["Przechwytywanie"],
)
async def get_capture_image(session_id: str, device_id: str, frame_index: int):
    """Serwuje pojedyncze zdjęcie pomiarowe."""
    session = await _get_or_404(session_id)
    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")
    path = session.capture_dir(device_id) / f"capture_{frame_index:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Zdjęcie {frame_index} nie istnieje")
    return FileResponse(path, media_type="image/png")


@app.delete(
    "/sessions/{session_id}/capture/images/{device_id}/{frame_index}",
    status_code=200,
    tags=["Przechwytywanie"],
)
async def delete_capture_frame(
    session_id: str,
    device_id: str,
    frame_index: int,
    requester_id: str = Query(..., description="ID lidera autoryzującego operację"),
):
    """Usuwa konkretne zdjęcie pomiarowe i aktualizuje licznik urządzenia."""
    session = await _get_or_404(session_id)
    requester = session.devices.get(requester_id)
    if requester is None:
        raise HTTPException(status_code=404, detail=f"Requester '{requester_id}' nie jest w sesji")
    if not requester.is_leader:
        raise HTTPException(status_code=403, detail="Tylko lider może usuwać zdjęcia pomiarowe")
    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    path = session.capture_dir(device_id) / f"capture_{frame_index:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Zdjęcie {frame_index} nie istnieje")
    path.unlink()

    remaining = list(session.capture_dir(device_id).glob("capture_*.png"))
    session.devices[device_id].capture_frame_count = len(remaining)
    await store.save(session_id)

    log.info("[%s] Usunięto zdjęcie pomiarowe %d urządzenia %s (przez: %s)",
             session_id, frame_index, device_id, requester_id)

    await ws_manager.broadcast(session_id, {
        "event": "capture_frame_deleted",
        "device_id": device_id,
        "frame_index": frame_index,
        "total_frames": session.devices[device_id].capture_frame_count,
    })
    return {
        "device_id": device_id,
        "frame_index": frame_index,
        "total_frames": session.devices[device_id].capture_frame_count,
    }


# ===========================================================================
# POMIAR
# ===========================================================================

@app.post(
    "/sessions/{session_id}/measure",
    status_code=202,
    tags=["Pomiar"],
)
async def run_measurement(session_id: str):
    """Uruchamia pelny pipeline pomiaru 3D w tle.

    Etapy:
      - Rektyfikacja stereo
      - Mapa dysparycji (SGBM)
      - Konwersja na chmurę punktów
      - Detekcja europalety (RANSAC)
      - Segmentacja i pomiar obiektu

    Wynik dostępny przez GET /sessions/{sid}/measurement i WebSocket
    (event: "measurement_done" lub "error").

    Wymagania:
    - Sesja w stanie READY (po kalibracji)
    - Co najmniej 1 zdjęcie pomiarowe per urządzenie
    """
    session = await _get_or_404(session_id)

    if session.state not in (SessionState.READY, SessionState.DONE):
        raise HTTPException(
            status_code=409,
            detail=f"Kalibracja wymagana (teraz: {session.state})",
        )

    if session.state == SessionState.PROCESSING:
        raise HTTPException(status_code=409, detail="Pipeline pomiaru już trwa")

    if session.min_capture_frames() < 1:
        raise HTTPException(
            status_code=409,
            detail="Brak zdjęć pomiarowych - prześlij zdjecia przez /capture/images",
        )

    await store.set_state(session_id, SessionState.PROCESSING)
    asyncio.create_task(measure_session(session_id))

    log.info("[%s] Pipeline pomiaru uruchomiony w tle", session_id)
    return {"message": "Pomiar uruchomiony", "state": SessionState.PROCESSING}


@app.get(
    "/sessions/{session_id}/measurement",
    response_model=MeasurementOut,
    tags=["Pomiar"],
)
async def get_measurement(session_id: str):
    """Zwraca wyniki ostatniego pomiaru (szerokosc, dlugosc, wysokosc w mm)."""
    session = await _get_or_404(session_id)

    if session.meas_result is None:
        raise HTTPException(
            status_code=404,
            detail="Brak wynikow pomiaru - uruchom POST /measure",
        )

    return _meas_to_out(session.meas_result)


@app.get(
    "/sessions/{session_id}/measurement/report",
    response_class=PlainTextResponse,
    tags=["Pomiar"],
)
async def get_measurement_report(session_id: str):
    """Zwraca pelny raport tekstowy ostatniego pomiaru."""
    session = await _get_or_404(session_id)

    if session.meas_result is None:
        raise HTTPException(status_code=404, detail="Brak wyników pomiaru")

    return PlainTextResponse(session.meas_result.report, media_type="text/plain; charset=utf-8")


# ===========================================================================
# POMIAR SYNTETYCZNY (test bez kamer)
# ===========================================================================

@app.post(
    "/measure/synthetic",
    response_model=MeasurementOut,
    tags=["Utility"],
)
async def measure_synthetic():
    """Uruchamia pelny pipeline na danych syntetycznych (bez kamer, do testow API).

    Generuje wirtualną scenę (3 pudełka na tle, baza 120 mm, f=800 px),
    przetwarza przez SGBM i zwraca pomiary wymiarów.

    Uwaga: Scena syntetyczna używa SGBM na płaskim jednolitym tle, więc
    detekcja palety może się nie udać - wtedy zwracany jest błąd 422.
    """
    try:
        result = await synthetic_measure()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return _meas_to_out(result)


# ===========================================================================
# WEBSOCKET
# ===========================================================================

async def _ws_keepalive(ws: WebSocket, interval: float = 20.0) -> None:
    """Sends periodic server-side pings to prevent proxy/NAT idle timeout."""
    while True:
        await asyncio.sleep(interval)
        try:
            await ws.send_json({"event": "ping"})
        except Exception:
            return


@app.websocket("/ws/{session_id}/{device_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str, device_id: str):
    """Kanal WebSocket dla synchronizacji urzadzen w czasie rzeczywistym.

    Protokol (client → server):
      {"action": "ping"}
        → {"event": "pong", "t": <timestamp>}

      {"action": "captured", "at": <timestamp>}
        → broadcast: {"event": "device_captured", "device_id": "...", "at": ...}

    Serwer → klienci (broadcast):
      {"event": "session_state",        <pelny SessionOut> }   ← tylko do laczacego urzadzenia
      {"event": "device_ws_connected",  "device_id": ...}
      {"event": "device_ws_disconnected","device_id": ...}
      {"event": "device_joined",        "device_id": ..., "is_leader": ...}
      {"event": "calibration_done",     "reproj_error": ...}
      {"event": "capture_trigger",      "at": ...}
      {"event": "measurement_done",     "width_mm": ..., "length_mm": ..., "height_mm": ..., "validation_passed": ...}
      {"event": "error",                "message": "..."}
    """
    # Walidacja sesji przed akceptacją polaczenia
    try:
        session = await store.get(session_id)
    except KeyError:
        await ws.close(code=4004, reason=f"Sesja '{session_id}' nie istnieje")
        return

    if device_id not in session.devices:
        await ws.close(code=4003, reason=f"Urządzenie '{device_id}' nie jest w sesji")
        return

    await ws_manager.connect(ws, session_id, device_id)
    session.devices[device_id].ws_connected = True
    log.info("[%s] WS połączono: %s", session_id, device_id)

    # Push current session state to this device so it catches up on any missed events
    await ws.send_json({"event": "session_state", **_session_to_out(session).model_dump()})

    # Tell all other devices that this device's WS is now live
    await ws_manager.broadcast(session_id, {
        "event": "device_ws_connected",
        "device_id": device_id,
    })

    keepalive_task = asyncio.create_task(_ws_keepalive(ws))
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "ping":
                await ws.send_json({"event": "pong", "t": time.time()})

            elif action == "captured":
                # Urządzenie potwierdza wykonanie zdjecia
                await ws_manager.broadcast(session_id, {
                    "event": "device_captured",
                    "device_id": device_id,
                    "at": data.get("at"),
                })

            # Nieznane akcje: ignoruj (forward-compatibility)

    except WebSocketDisconnect:
        pass

    except Exception as exc:
        log.warning("[%s] WS blad %s: %s", session_id, device_id, exc)

    finally:
        keepalive_task.cancel()
        ws_manager.disconnect(session_id, device_id)
        if device_id in session.devices:
            session.devices[device_id].ws_connected = False
        log.info("[%s] WS rozłączono: %s", session_id, device_id)

        # Notify remaining devices that this device's WS dropped
        await ws_manager.broadcast(session_id, {
            "event": "device_ws_disconnected",
            "device_id": device_id,
        })
