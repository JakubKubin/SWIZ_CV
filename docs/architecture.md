# Architektura systemu

## Diagram komponentów

```
┌──────────────────────────────────────────────────────────┐
│                    Urządzenia mobilne                     │
│                                                          │
│  ┌─────────────────┐        ┌─────────────────┐          │
│  │   Flutter App   │        │   Flutter App   │          │
│  │   (Leader)      │        │   (Follower)    │          │
│  │   Lewa kamera   │        │   Prawa kamera  │          │
│  └────────┬────────┘        └────────┬────────┘          │
└───────────┼─────────────────────────┼────────────────────┘
            │  REST API + WebSocket   │
            ▼                         ▼
┌──────────────────────────────────────────────────────────┐
│                    Backend (Docker)                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  FastAPI (backend/main.py)                       │    │
│  │  - /sessions/*    - /calibration/*               │    │
│  │  - /capture/*     - /measurement/*               │    │
│  │  - WS /ws/{sid}/{device_id}                      │    │
│  └──────────────────┬───────────────────────────────┘    │
│                     │ asyncio.to_thread()                 │
│  ┌──────────────────▼───────────────────────────────┐    │
│  │  Pipeline CV (wątki robocze)                     │    │
│  │                                                  │    │
│  │  calibration.py → disparity.py → pointcloud.py  │    │
│  │  pallet.py      → measurement.py                │    │
│  └──────────────────┬───────────────────────────────┘    │
│                     │                                     │
│  ┌──────────────────▼───────────────────────────────┐    │
│  │  Storage (Docker Volume: /app/data)              │    │
│  │  data/{session_id}/                              │    │
│  │  ├── calib/{device_id}/frame_NNNN.jpg           │    │
│  │  ├── captures/{device_id}/capture_NNNN.jpg      │    │
│  │  ├── stereo.json    (parametry kalibracji)       │    │
│  │  ├── cloud.ply      (chmura punktów)             │    │
│  │  └── measurement_report.txt                     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Maszyna stanów sesji

Sesja przechodzi przez następujące stany od momentu utworzenia do wyniku pomiaru:

```
                    POST /sessions
                         │
                         ▼
                      ┌──────┐
                      │ IDLE │  ← Oczekiwanie na 2 urządzenia
                      └──┬───┘
                         │ POST /calibration/compute
                         ▼
                  ┌─────────────┐
                  │ CALIBRATING │  ← Obliczenia Zhang (wątek)
                  └──────┬──────┘
               sukces ───┤─── błąd
                    ┌────┘    └──→ IDLE (możliwy retry)
                    ▼
                 ┌───────┐
                 │ READY │  ← Kalibracja OK, gotowy do pomiaru
                 └───┬───┘
                     │ POST /measure
                     ▼
              ┌────────────┐
              │ PROCESSING │  ← Pipeline 3D (wątek)
              └──────┬─────┘
           sukces ───┤─── błąd
                ┌────┘    └──→ READY (możliwy retry)
                ▼
             ┌──────┐
             │ DONE │  ← Wyniki dostępne przez GET /measurement
             └──────┘
```

**Przejścia przy błędzie:**
- `CALIBRATING → IDLE` — kalibracja nieudana (zbyt mało klatek, zły wzorzec)
- `PROCESSING → READY` — pomiar nieudany (można spróbować ponownie)

**Usunięcie urządzenia:** Jeśli jedno urządzenie odłączy się podczas sesji, wymagana jest ponowna kalibracja od początku.

---

## Przepływ danych — od zdjęcia do pomiaru

```
[Flutter Leader klika „Trigger"]
        │
        ├─ POST /sessions/{sid}/capture/trigger
        │  Serwer: Target_Timestamp = now() + delay_ms
        │  WebSocket broadcast → {"event": "capture_trigger", "at": T}
        │
        ├─ Flutter Leader: wykonaj zdjęcie o czasie T
        ├─ Flutter Follower: wykonaj zdjęcie o czasie T
        │
        ├─ POST /sessions/{sid}/capture/images (Leader)   → left/capture_0000.jpg
        ├─ POST /sessions/{sid}/capture/images (Follower) → right/capture_0000.jpg
        │
        └─ POST /sessions/{sid}/measure
           │
           └─ asyncio.to_thread(_sync_measure)
              │
              ├─ load_params(stereo.json)       → StereoParams
              ├─ rectify_pair(left, right)       → left_rect, right_rect
              ├─ compute_disparity(SGBM)         → disp (px)
              ├─ build_pointcloud(disp, Q)       → xyz [mm], colors
              ├─ filter_pointcloud()             → xyz (bez outlierów)
              ├─ detect_pallet(xyz)             → płaszczyzna, układ palety
              ├─ measure_object()               → bbox (W, L, H)
              ├─ validate_measurement()         → ValidationReport
              └─ generate_report()             → tekst
                     │
                     └─ WebSocket broadcast → {"event": "measurement_done", ...}
                        GET /sessions/{sid}/measurement → MeasurementOut JSON
```

---

## Synchronizacja zdjęć (Precision Sync)

Problem: dwie kamery na różnych urządzeniach muszą wyzwolić migawkę w tym samym momencie, mimo różnic w opóźnieniu sieciowym (network jitter).

Rozwiązanie — **Buffered Delayed Capture**:

1. Leader klika przycisk → `POST /capture/trigger` z `delay_ms=500` (domyślnie)
2. Serwer wylicza `target_ts = unix_now() + 0.5s`
3. WebSocket broadcast do wszystkich urządzeń: `{"event": "capture_trigger", "at": target_ts}`
4. Każde urządzenie czeka do `target_ts` i wyzwala migawkę dokładnie w tym czasie
5. Błąd synchronizacji ≈ jitter sieci (~10–50 ms), znacznie mniej niż bez synchronizacji

---

## Komunikacja sieciowa

| Kanał | Technologia | Zastosowanie |
|-------|-------------|--------------|
| REST API | HTTP/JSON | Zarządzanie sesją, upload plików, polling wyników |
| WebSocket | `ws://host/ws/{sid}/{dev}` | Trigger zdjęcia, broadcast wyników, heartbeat |

Serwer **nie inicjuje** HTTP do klientów — wszystkie push-y idą przez WebSocket.
