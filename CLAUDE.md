# Stereo Vision - Pomiar obiektow na europalecie

## Cel projektu
System stereowizyjny do pomiaru wymiarów (długość, szerokość, wysokość) obiektów umieszczonych na standardowej europalecie (1200x800 mm). Dwa telefony na statywach (50-100 cm od obiektu) działają jako para stereo. Aplikacja Flutter zarządza sesją, kalibruje kamery metodą Zhanga, synchronicznie wyzwala zdjęcia i wyświetla wyniki. Backend FastAPI wykonuje cały pipeline 3D.

## Stan implementacji
**Projekt jest w pełni zaimplementowany.** Wszystkie fazy zostały ukończone.

## Architektura systemu (faktyczna)

### Komponenty
| Komponent | Technologia | Stan |
|-----------|------------|------|
| Orkiestracja | Docker Compose | ✅ |
| API + WebSocket | FastAPI + Uvicorn | ✅ |
| Zadania w tle | `asyncio.to_thread()` (nie Celery) | ✅ |
| Stan sesji | In-memory + zapis JSON na dysk (nie Redis) | ✅ |
| Storage | Docker volume `session_data:/app/data` | ✅ |
| Frontend | Flutter (poza Dockerem) | ✅ |

> **Uwaga:** Finalna architektura nie używa Redis ani Celery. Sesje trzymane są w pamięci z persystencją na dysk (`data/{session_id}/session.json`). Zadania CPU-intensive wykonywane przez `asyncio.to_thread()`.

### Założenia sprzętowe
- Docelowo dwa urządzenia (kod obsługuje więcej)
- Lider = lewa kamera, Follower = prawa kamera
- Flutter działa na Androidzie/iOS — nie jest w Dockerze
- Kalibracja per para MAC adresów urządzeń
- Rozłączenie urządzenia = wymagana rekalibracja

## Struktura folderów (faktyczna)

```
swiz/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── env.example
│
├── calibration.py        # Kalibracja Zhang (single + stereo), JSON I/O
├── disparity.py          # SGBM, rektyfikacja, konwersja disparity->depth
├── pointcloud.py         # Budowa chmury punktów, filtracja, zapis PLY
├── pallet.py             # Detekcja palety RANSAC+SVD, transformacja, ROI
├── measurement.py        # Segmentacja, bbox 3D, 3x estymacja obj., walidacja
├── pipeline.py           # Orkiestrator 8-etapowego pipeline (tryb real + synthetic)
├── config.py             # Centralna konfiguracja z .env
├── logging_setup.py      # Logging: konsola (INFO) + plik rotacyjny (DEBUG)
│
├── backend/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, 30+ endpointów REST + WebSocket
│   ├── schemas.py        # Modele Pydantic (request/response)
│   ├── session.py        # SessionStore, state machine, SessionState enum
│   └── tasks.py          # Zadania w tle + WebSocketManager
│
├── flutter_app/          # Aplikacja mobilna/web
│   └── lib/
│       ├── main.dart
│       ├── providers/
│       │   └── app_state.dart        # Provider: globalny stan aplikacji
│       ├── services/
│       │   └── api_service.dart      # Klient HTTP + WebSocket
│       ├── models/
│       │   └── models.dart           # Data classes (mirror Pydantic schemas)
│       ├── theme/
│       │   └── app_theme.dart
│       ├── screens/
│       │   ├── home_screen.dart
│       │   ├── session_screen.dart
│       │   ├── calibration_screen.dart
│       │   ├── capture_screen.dart
│       │   └── results_screen.dart
│       ├── widgets/
│       │   └── app_banner.dart
│       └── utils/
│           └── log.dart
│
├── test_calibration.py   # ~30 testów kalibracji (synthetic + projections)
├── test_measurement.py   # ~40 testów pomiaru (RANSAC, bbox, walidacja)
├── conftest.py
│
└── data/                 # Docker volume
    └── {session_id}/
        ├── session.json
        ├── calib/{device_id}/    # Zdjęcia szachownicy
        ├── captures/{device_id}/ # Zdjęcia pomiarowe
        ├── stereo.json           # Parametry kalibracji stereo
        ├── cloud.ply
        └── measurement_report.txt
```

## API Endpoints (faktyczne)

### Sesje
| Metoda | Endpoint | Opis |
|--------|----------|------|
| POST | `/sessions` | Utwórz nową sesję (zwraca session_id) |
| GET | `/sessions/{id}` | Stan sesji + lista urządzeń |
| GET | `/sessions` | Lista wszystkich aktywnych sesji |
| DELETE | `/sessions/{id}` | Usuń sesję i dane |
| POST | `/sessions/{id}/join` | Dołącz urządzenie (`device_id`, MAC, `is_leader`) |
| DELETE | `/sessions/{id}/devices/{device_id}` | Opuść sesję (dane zachowane) |

### Kalibracja
| Metoda | Endpoint | Opis |
|--------|----------|------|
| POST | `/sessions/{id}/calibration/images` | Upload zdjęcia szachownicy (multipart, auto-numeracja) |
| POST | `/sessions/{id}/calibration/compute` | Uruchom kalibrację w tle |
| GET | `/sessions/{id}/calibration` | Status kalibracji + RMS error |

### Akwizycja
| Metoda | Endpoint | Opis |
|--------|----------|------|
| POST | `/sessions/{id}/capture/trigger` | Broadcast TRIGGER z Target_Timestamp (NTP sync); opcjonalny param `delay_ms` |
| POST | `/sessions/{id}/capture/images` | Upload zdjęcia pomiarowego (multipart) |

### Pomiar i wyniki
| Metoda | Endpoint | Opis |
|--------|----------|------|
| POST | `/sessions/{id}/measure` | Uruchom pipeline 3D w tle |
| GET | `/sessions/{id}/measurement` | Wyniki pomiaru (W/L/H mm, 3x obj., walidacja) |
| GET | `/sessions/{id}/measurement/report` | Pełny raport tekstowy |
| POST | `/measure/synthetic` | Test pipeline na danych syntetycznych (bez kamer) |

### Inne
| Metoda | Endpoint | Opis |
|--------|----------|------|
| GET | `/health` | Health check |
| WS | `/ws/{session_id}/{device_id}` | WebSocket (eventy, sync, heartbeat) |

## State Machine sesji

```
IDLE → CALIBRATING → READY → PROCESSING → DONE
         ↑_____________↑         ↑___________↑
         (błąd kalibracji)       (błąd pomiaru, retry)
```

- **IDLE**: Oczekiwanie na urządzenia
- **CALIBRATING**: Zbieranie zdjęć szachownicy + obliczenia Zhang
- **READY**: Parametry stereo gotowe, system gotowy do pomiaru
- **PROCESSING**: Serwer przetwarza chmurę punktów
- **DONE**: Wyniki dostępne

## Protokół Precision Sync (akwizycja)

1. Przy WebSocket handshake: klient+serwer wyznaczają NTP offset
2. Lider klika trigger → serwer oblicza `Target_Timestamp = now + 1000ms`
3. Serwer broadcastuje `{"action": "capture", "at": Target_Timestamp}` do wszystkich
4. Każde urządzenie planuje zdjęcie na `Target_Timestamp - local_offset`

## Pipeline 3D (8 etapów)

| Etap | Moduł | Opis |
|------|-------|------|
| 1 | `calibration.py` | Kalibracja Zhang: K, dist, R, T, E, F, R1/R2/P1/P2/Q |
| 2 | Backend/Flutter | Synchroniczna akwizycja par stereo |
| 3 | `disparity.py` | Rektyfikacja (remap do układu równoległego) |
| 4 | `disparity.py` | SGBM + WLS filter → mapa głębi (mm) via Q matrix |
| 5 | `pointcloud.py` | Budowa chmury XYZ, filtracja statystyczna (k-NN) |
| 6 | `pallet.py` | RANSAC (1000 iter) + SVD → płaszczyzna palety, ROI 1200×800 mm |
| 7 | `measurement.py` | Segmentacja: noise floor 20 mm, bbox 3D |
| 8 | `measurement.py` | Trzy estymacje obj.: voxel (najdokładniejszy), bbox, hull (scipy opcjonalny) |

## Zmienne środowiskowe (.env)

```env
# Szachownica (domyślne z config.py w nawiasach)
CHECKERBOARD_ROWS=7          # Wewnętrzne narożniki (wiersze) [default: 5]
CHECKERBOARD_COLS=9          # Wewnętrzne narożniki (kolumny) [default: 8]
SQUARE_SIZE_MM=44.0          # Rozmiar kwadratu w mm [default: 15.0]

# Ścieżki
CALIBRATION_DIR=/app/data/calib
CALIBRATION_OUTPUT=/app/data/calib_output

# Serwer
BACKEND_PORT=8000

# Opcjonalne (wartości domyślne w config.py)
LOG_LEVEL=INFO
CORNER_DETECT_MAX_WIDTH=1920  # Skalowanie dla wysokich rozdzielczości (telefony) [default: 1920]
MIN_CALIBRATION_IMAGES=3
MAX_STEREO_REPROJ_ERROR=2.0   # px; powyżej → ostrzeżenie
```

Stałe nieeksponowane przez .env (hardcoded w config.py):
- `PALLET_HEIGHT_MM = 144.0` — standardowa wysokość europalety

## Testy

```bash
pytest test_calibration.py   # ~30 testów: Zhang, stereo, serializacja, wysokie rozdzielczości
pytest test_measurement.py   # ~40 testów: RANSAC, bbox, obj., walidacja
```

Testy używają danych syntetycznych (idealne projekcje 3D) — nie wymagają kamer.
`conftest.py` dodaje flagę `--visualize` do zapisu obrazów diagnostycznych.

## Uruchomienie

```bash
# Backend
docker compose up --build

# Flutter (lokalnie)
cd flutter_app
flutter run                  # Android/iOS/web
```

Backend dostępny na `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

## Kluczowe decyzje implementacyjne

- **Brak Redis/Celery**: Sesje in-memory z persystencją JSON; zadania przez `asyncio.to_thread()`. Wystarczające przy 2 urządzeniach i jednej parze stereo na raz.
- **Parametry kalibracji per para MAC**: `data/{session_id}/stereo.json` identyfikowany przez MAC adresy obu urządzeń.
- **scipy opcjonalny**: Estymacja hull (`hull_mm3`) dostępna tylko jeśli scipy zainstalowane; pozostałe dwie metody zawsze dostępne.
- **Skalowanie wysokich rozdzielczości**: Detekcja narożników na przeskalowanym obrazie (`CORNER_DETECT_MAX_WIDTH`), refinement na oryginalnym — dla kamer 12+ Mpx.
- **CORS**: Aktualnie `allow_origins=["*"]` — do ograniczenia w produkcji.
- **Disconnect = rekalibracja**: Brak mechanizmu reconnect; zerwanie WebSocket wymaga ponownej kalibracji.
