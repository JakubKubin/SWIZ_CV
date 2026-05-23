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
import logging
import time

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form,
    WebSocket, WebSocketDisconnect, Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .schemas import (
    JoinRequest, SessionOut, DeviceOut,
    CalibStatusOut, TriggerRequest, TriggerOut,
    MeasurementOut, HealthOut,
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


def _session_to_out(session) -> SessionOut:
    """Konwertuje obiekt Session na Pydantic SessionOut."""
    return SessionOut(
        session_id=session.session_id,
        state=session.state,
        devices=[
            DeviceOut(
                device_id=d.device_id,
                mac=d.mac,
                is_leader=d.is_leader,
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
        raise HTTPException(status_code=409, detail="Urządzenie już dołączyło do sesji")

    if session.is_full():
        raise HTTPException(status_code=409, detail="Sesja jest pełna (maks. 2 urządzenia)")

    # Tylko jedno urzadzenie moze byc leaderem
    if body.is_leader and session.leader() is not None:
        raise HTTPException(status_code=409, detail="Leader już istnieje w tej sesji")

    from .session import Device
    device = Device(
        device_id=body.device_id,
        mac=body.mac,
        is_leader=body.is_leader,
    )
    session.devices[body.device_id] = device
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
async def leave_session(session_id: str, device_id: str):
    """Wypisuje jedno urządzenie z sesji (np. przy wyjściu z aplikacji).

    Sesja oraz jej dane (kalibracja, zdjęcia, wyniki) NIE są usuwane - dzięki
    temu użytkownik może później wrócić do sesji i ponownie dołączyć tym samym
    urządzeniem. Trwałe usunięcie wykonuje się jawnie przez DELETE /sessions/{id}.
    """
    session = await _get_or_404(session_id)

    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    del session.devices[device_id]
    await store.save(session_id)
    log.info("[%s] Urządzenie opuściło sesję: %s (%d pozostało)",
             session_id, device_id, len(session.devices))

    await ws_manager.broadcast(session_id, {
        "event": "device_left",
        "device_id": device_id,
        "remaining": len(session.devices),
    })

    return Response(status_code=204)


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

    Obrazy sa numerowane sekwencyjnie: frame_0000.jpg, frame_0001.jpg, ...
    Wywołaj co najmniej 3 razy dla każdego urządzenia przed uruchomieniem kalibracji.
    """
    session = await _get_or_404(session_id)

    if device_id not in session.devices:
        raise HTTPException(status_code=404, detail=f"Urządzenie '{device_id}' nie jest w sesji")

    device = session.devices[device_id]
    calib_dir = session.calib_dir(device_id)
    calib_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".jpg"  # normalizuj do jpg
    save_path = calib_dir / f"frame_{device.calib_frame_count:04d}{suffix}"

    content = await file.read()
    save_path.write_bytes(content)

    device.calib_frame_count += 1
    await store.save(session_id)

    log.info("[%s] Kalibracja %s: klatka %d zapisana (%d B)",
             session_id, device_id, device.calib_frame_count - 1, len(content))

    return {
        "device_id": device_id,
        "frame_index": device.calib_frame_count - 1,
        "total_frames": device.calib_frame_count,
    }


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

    if len(session.devices) < 2:
        raise HTTPException(
            status_code=409,
            detail="Potrzeba 2 urządzeń (leader + follower) do kalibracji stereo",
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

    Obrazy są numerowane sekwencyjnie: capture_0000.jpg, capture_0001.jpg, ...
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

    save_path = capture_dir / f"capture_{device.capture_frame_count:04d}.jpg"
    content = await file.read()
    save_path.write_bytes(content)

    device.capture_frame_count += 1
    await store.save(session_id)

    log.info("[%s] Zdjecie %s: capture %d (%d B)",
             session_id, device_id, device.capture_frame_count - 1, len(content))

    return {
        "device_id": device_id,
        "frame_index": device.capture_frame_count - 1,
        "total_frames": device.capture_frame_count,
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

@app.websocket("/ws/{session_id}/{device_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str, device_id: str):
    """Kanal WebSocket dla synchronizacji urzadzen w czasie rzeczywistym.

    Protokol (client → server):
      {"action": "ping"}
        → {"event": "pong", "t": <timestamp>}

      {"action": "captured", "at": <timestamp>}
        → broadcast: {"event": "device_captured", "device_id": "...", "at": ...}

    Serwer → klienci (broadcast):
      {"event": "device_joined",       "device_id": ..., "is_leader": ...}
      {"event": "calibration_done",    "reproj_error": ...}
      {"event": "capture_trigger",     "at": ...}
      {"event": "measurement_done",    "width_mm": ..., "length_mm": ..., "height_mm": ..., "validation_passed": ...}
      {"event": "error",               "message": "..."}
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
        ws_manager.disconnect(session_id, device_id)
        if device_id in session.devices:
            session.devices[device_id].ws_connected = False
        log.info("[%s] WS rozłączono: %s", session_id, device_id)
