# Backend — FastAPI

Backend to asynchroniczny serwer Python oparty na FastAPI. Wchodzi w Dockerze na porcie 8000.

## Pliki

| Plik | Odpowiedzialność |
|------|-----------------|
| [backend/main.py](../backend/main.py) | Definicje wszystkich endpointów HTTP i WebSocket |
| [backend/session.py](../backend/session.py) | Maszyna stanów sesji, model urządzenia, SessionStore |
| [backend/tasks.py](../backend/tasks.py) | Zadania w tle (kalibracja, pomiar) + WSManager |
| [backend/schemas.py](../backend/schemas.py) | Modele Pydantic — walidacja request/response |

---

## session.py — Maszyna stanów

### SessionState (enum)

```python
IDLE        # sesja założona, oczekuje na urządzenia
CALIBRATING # kalibracja w toku
READY       # kalibracja OK — gotowy do pomiaru
PROCESSING  # pipeline 3D w toku
DONE        # wyniki dostępne
```

### Device (dataclass)

Reprezentuje jedno podłączone urządzenie:

| Pole | Typ | Opis |
|------|-----|------|
| `device_id` | `str` | UUID urządzenia |
| `mac` | `str` | Adres MAC (identyfikator fizyczny) |
| `is_leader` | `bool` | `True` = lewa kamera (inicjuje triggery) |
| `ws_connected` | `bool` | Czy WebSocket jest aktywny |
| `calib_frame_count` | `int` | Liczba przesłanych klatek kalibracyjnych |
| `capture_frame_count` | `int` | Liczba przesłanych zdjęć pomiarowych |

### Session

Sesja grupuje dwoje urządzeń i trzyma ścieżki do danych na dysku:

```
data/{session_id}/
├── calib/{device_id}/frame_NNNN.jpg    ← obrazy kalibracyjne
├── captures/{device_id}/capture_NNNN.jpg ← zdjęcia pomiarowe
└── stereo.json                          ← wynik kalibracji
```

Pomocnicze metody:
- `session.leader()` — zwraca urządzenie z `is_leader=True`
- `session.follower()` — zwraca pierwsze urządzenie z `is_leader=False`
- `session.min_calib_frames()` — minimum klatek kalibracyjnych spośród urządzeń (bottleneck)

### SessionStore

Globalny singleton (`store = SessionStore()`) przechowujący sesje w pamięci operacyjnej.
Dostęp asynchroniczny przez `asyncio.Lock`. Nie używa Redisa — wystarczy do aktualnej skali.

**Persystencja:** metadane każdej sesji (stan, urządzenia, wyniki kalibracji i pomiaru)
są zapisywane do `data/{session_id}/session.json` po każdej mutacji (utworzenie, dołączenie,
opuszczenie, upload klatki, zmiana stanu). Przy starcie `SessionStore` wczytuje wszystkie
zapisane sesje z dysku, dzięki czemu użytkownicy mogą wrócić do swoich sesji po restarcie
backendu. Pole `ws_connected` jest transientowe (po wczytaniu zawsze `False`). Dane są
usuwane wyłącznie jawnie przez `DELETE /sessions/{id}` — opuszczenie sesji ich nie kasuje.

---

## tasks.py — Zadania w tle

Zadania CPU-intensive (OpenCV) **nie mogą** blokować pętli zdarzeń uvicorn.
Rozwiązanie: `asyncio.to_thread(funkcja_synchroniczna)` — uruchamia obliczenia w puli wątków.

### WSManager

Zarządza połączeniami WebSocket per `(session_id, device_id)`.

```python
await ws_manager.connect(ws, session_id, device_id)  # rejestruj
ws_manager.disconnect(session_id, device_id)          # wyrejestruj
await ws_manager.broadcast(session_id, payload)       # wyślij do wszystkich w sesji
await ws_manager.send(session_id, device_id, payload) # wyślij do konkretnego
```

### calibrate_session(session_id)

Async wrapper — uruchamia `_sync_calibrate` w wątku:

1. Pobiera sesję (`store.get_sync`)
2. Wczytuje ścieżki klatek kalibracyjnych (leader + follower)
3. Wywołuje `calibrate_stereo()` z [calibration.py](../calibration.py)
4. Zapisuje wynik do `stereo.json`
5. Ustawia stan sesji na `READY`
6. Broadcast WebSocket: `{"event": "calibration_done", "reproj_error": ...}`

Przy błędzie: stan → `IDLE`, broadcast `{"event": "error", "message": ...}`

### measure_session(session_id)

Async wrapper — uruchamia `_sync_measure` w wątku:

1. Wczytuje `stereo.json` → `StereoParams`
2. Wczytuje najnowsze zdjęcia z `captures/{device_id}/`
3. Skaluje obrazy > 1920px do 1920px
4. `rectify_pair()` → `left_rect, right_rect`
5. `compute_disparity(SGBM)` → mapa dysparycji
6. `build_pointcloud()` → `xyz [mm]`, `colors`
7. `filter_pointcloud()` — usunięcie outlierów
8. `detect_pallet(xyz)` → `PalletDetectionResult`
9. `measure_object()` → `MeasurementResult` (bbox W/L/H)
10. `validate_measurement()`, `generate_report()`
11. Zapis `cloud.ply` i `measurement_report.txt`
12. Stan → `DONE`, broadcast `{"event": "measurement_done", ...}`

Przy błędzie: stan → `READY` (umożliwia retry).

---

## schemas.py — Modele Pydantic

### Wejście

```python
JoinRequest:
  device_id: str        # UUID urządzenia
  mac: str              # adres MAC
  is_leader: bool = False

TriggerRequest:
  delay_ms: int = 500   # opóźnienie wyzwalania [ms], 0–10000
```

### Wyjście

```python
SessionOut:
  session_id: str
  state: str            # IDLE / CALIBRATING / READY / PROCESSING / DONE
  devices: list[DeviceOut]
  created_at: float     # Unix timestamp
  has_calibration: bool # czy zapisano parametry kalibracji
  has_measurement: bool # czy zapisano wynik pomiaru

DeviceOut:
  device_id, mac, is_leader, joined_at
  ws_connected: bool
  calib_frame_count: int
  capture_frame_count: int

CalibStatusOut:
  state: str
  reproj_error: float | None   # błąd reprojekcji [px], None przed kalibracją
  message: str

TriggerOut:
  at: float             # Unix timestamp momentu wyzwolenia
  delay_ms: int

MeasurementOut:
  validation_passed: bool
  width_mm, length_mm, height_mm: float
  volume_voxel_mm3: float      # objętość metodą kolumnową (height-field)
  volume_bbox_mm3: float       # objętość bounding box (W×L×H)
  volume_hull_mm3: float|null  # objętość convex hull (null gdy niedostępna)
  fill_ratio: float            # voxel / bbox — "pełność" bryły [0..1]
  pallet_rms_mm: float        # residuum płaszczyzny palety [mm]
  n_object_pts: int            # liczba punktów obiektu
  n_pallet_inliers: int        # liczba inlierów RANSAC palety
  issues: list[str]            # lista ostrzeżeń/błędów walidacji
  report: str                  # pełny raport tekstowy

HealthOut:
  status: str = "ok"
  version: str = "1.0.0"
```

---

## Obsługa błędów

| Kod | Sytuacja |
|-----|---------|
| 404 | Sesja lub urządzenie nie istnieje |
| 409 | Konflikt stanu (np. kalibracja już trwa, sesja pełna) |
| 422 | Błąd pipeline (np. brak kamer → tryb syntetyczny) |
| WS close 4003 | Urządzenie nie jest w sesji |
| WS close 4004 | Sesja nie istnieje |

---

## Swagger UI

Po uruchomieniu serwera interaktywna dokumentacja dostępna pod:

```
http://localhost:8000/docs
```
