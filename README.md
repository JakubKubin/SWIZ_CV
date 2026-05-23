# Stereo Vision — Pomiar obiektów na europalecie

System stereowizyjny mierzący wymiary (długość × szerokość × wysokość) obiektów umieszczonych na europalecie (1200 × 800 mm). Dwa smartfony na statywach działają jako para stereo. Aplikacja Flutter zarządza sesją, kalibruje kamery i synchronicznie wyzwala zdjęcia. Backend FastAPI wykonuje pełen pipeline 3D — od rektyfikacji po bounding box.

---

## Szybki start

### Wymagania

- Docker + Docker Compose
- Flutter SDK (do uruchomienia aplikacji mobilnej)

### 1. Sklonuj i skonfiguruj

```bash
git clone <repo-url>
cd swiz
cp env.example .env
```

Edytuj `.env` dopasowując do swojej szachownicy kalibracyjnej:

```env
CHECKERBOARD_ROWS=5      # wewnętrzne narożniki (wiersze)
CHECKERBOARD_COLS=8      # wewnętrzne narożniki (kolumny)
SQUARE_SIZE_MM=15.0      # fizyczny rozmiar kwadratu [mm]
BACKEND_PORT=8000
```

### 2. Uruchom backend

```bash
docker compose up --build
```

Backend dostępny na `http://<adres-ip-komputera>:8000`
Interaktywne docs API: `http://<adres-ip-komputera>:8000/docs`

### 3. Uruchom aplikację Flutter

```bash
cd flutter_app
flutter pub get
flutter run          # Android / iOS / web
```

W ustawieniach aplikacji wpisz adres IP komputera z backendem.

### 4. Przetestuj pipeline (bez kamer)

```bash
# lokalnie
python pipeline.py

# przez API
curl -X POST http://localhost:8000/measure/synthetic
```

---

## Architektura

```
┌─────────────────────────────────────────────┐
│              Flutter App (mobile/web)        │
│  home → session → calibration → capture → results │
│  Provider state │ HTTP + WebSocket           │
└───────────────────┬─────────────────────────┘
                    │ REST + WS
┌───────────────────▼─────────────────────────┐
│           FastAPI Backend (Docker)           │
│  /sessions  /calibration  /capture  /measure │
│  WebSocketManager │ asyncio.to_thread()      │
└───────────────────┬─────────────────────────┘
                    │ wywołania synchroniczne
┌───────────────────▼─────────────────────────┐
│             Pipeline 3D (Python)             │
│  calibration → disparity → pointcloud       │
│  pallet detection → measurement             │
└─────────────────────────────────────────────┘
```

### Stos technologiczny

| Warstwa | Technologia | Rola |
|---------|------------|------|
| Orkiestracja | Docker Compose | Jeden kontener — backend + data volume |
| API | FastAPI + Uvicorn | REST + WebSocket |
| Zadania w tle | `asyncio.to_thread()` | Kalibracja i pomiar bez blokowania event loop |
| Persystencja sesji | JSON na dysk | `data/{session_id}/session.json` — brak Redis |
| Vision pipeline | OpenCV + NumPy | Kalibracja, SGBM, chmura punktów, RANSAC |
| Volume estimation | scipy (opcjonalny) | ConvexHull; bez scipy — dwie pozostałe metody |
| Frontend | Flutter + Provider | Android / iOS / web z jednej bazy kodu |

---

## Workflow użytkownika

```
1. Lider tworzy sesję         POST /sessions
2. Follower dołącza           POST /sessions/{id}/join
3. Oba telefony otwierają WS  WS /ws/{id}/{device_id}
4. Wykonanie zdjęć szachownicy → upload × N klatek
5. Obliczenie kalibracji      POST /sessions/{id}/calibration/compute
6. Lider wyzwala zdjęcie      POST /sessions/{id}/capture/trigger
   └─ serwer broadcastuje Target_Timestamp (now + 1000 ms)
   └─ oba telefony strzelają zdjęcie dokładnie na T
7. Upload zdjęć pomiarowych   POST /sessions/{id}/capture/images
8. Uruchomienie pomiaru       POST /sessions/{id}/measure
9. Odczyt wyników             GET  /sessions/{id}/measurement
```

---

## Stan sesji

```
IDLE ──► CALIBRATING ──► READY ──► PROCESSING ──► DONE
              │                         │
              └──── błąd: powrót ───────┘
                    do poprzedniego stanu (retry)
```

---

## Pipeline 3D

| Etap | Moduł | Co robi |
|------|-------|---------|
| 1 | `calibration.py` | Zhang: K, dist, R, T, E, F, R1/R2/P1/P2/Q → `stereo.json` |
| 2 | Flutter | Synchroniczna akwizycja pary stereo (NTP offset) |
| 3 | `disparity.py` | Rektyfikacja — remap do układu równoległego |
| 4 | `disparity.py` | SGBM + WLS filter → mapa głębi w mm (Q matrix) |
| 5 | `pointcloud.py` | Chmura XYZ, filtracja statystyczna k-NN → `cloud.ply` |
| 6 | `pallet.py` | RANSAC (1000 iter) + SVD → płaszczyzna palety, ROI 1200×800 mm |
| 7 | `measurement.py` | Segmentacja: noise floor 20 mm, kontur, bbox 3D |
| 8 | `measurement.py` | Trzy estymacje objętości: voxel / bbox / hull (scipy) |

---

## Struktura projektu

```
swiz/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── env.example
│
├── calibration.py        # Kalibracja Zhang (single + stereo), JSON I/O
├── disparity.py          # SGBM, rektyfikacja, konwersja disparity→depth
├── pointcloud.py         # Chmura punktów, filtracja, zapis PLY
├── pallet.py             # RANSAC+SVD, transformacja do układu palety, ROI
├── measurement.py        # Segmentacja, bbox 3D, objętość, walidacja, raport
├── pipeline.py           # Orkiestrator pipeline (tryb real + synthetic)
├── config.py             # Centralna konfiguracja z .env
├── logging_setup.py      # Konsola INFO + plik rotacyjny DEBUG
│
├── backend/
│   ├── main.py           # FastAPI — wszystkie endpointy + WebSocket
│   ├── schemas.py        # Pydantic models (request / response)
│   ├── session.py        # SessionStore, SessionState enum, persystencja
│   └── tasks.py          # calibrate_session, measure_session, WebSocketManager
│
├── flutter_app/lib/
│   ├── main.dart
│   ├── providers/app_state.dart      # Globalny stan (Provider)
│   ├── services/api_service.dart     # HTTP + WebSocket klient
│   ├── models/models.dart            # Data classes
│   ├── theme/app_theme.dart
│   ├── screens/
│   │   ├── home_screen.dart
│   │   ├── session_screen.dart
│   │   ├── calibration_screen.dart
│   │   ├── capture_screen.dart
│   │   └── results_screen.dart
│   ├── widgets/app_banner.dart
│   └── utils/log.dart
│
├── test_calibration.py   # ~30 testów (Zhang, stereo, serializacja, hi-res)
├── test_measurement.py   # ~40 testów (RANSAC, bbox, objętość, walidacja)
└── conftest.py           # flaga --visualize
```

---

## API — przegląd endpointów

### Sesje
| Metoda | Endpoint | Opis |
|--------|----------|------|
| `POST` | `/sessions` | Utwórz sesję → `session_id` |
| `GET` | `/sessions` | Lista aktywnych sesji |
| `GET` | `/sessions/{id}` | Stan + lista urządzeń |
| `DELETE` | `/sessions/{id}` | Usuń sesję i dane |
| `POST` | `/sessions/{id}/join` | Dołącz urządzenie (`device_id`, MAC, `is_leader`) |
| `DELETE` | `/sessions/{id}/devices/{device_id}` | Opuść sesję |

### Kalibracja
| Metoda | Endpoint | Opis |
|--------|----------|------|
| `POST` | `/sessions/{id}/calibration/images` | Upload zdjęcia szachownicy (multipart) |
| `POST` | `/sessions/{id}/calibration/compute` | Uruchom kalibrację w tle |
| `GET` | `/sessions/{id}/calibration` | Status + RMS error |

### Akwizycja
| Metoda | Endpoint | Opis |
|--------|----------|------|
| `POST` | `/sessions/{id}/capture/trigger` | Broadcast Target_Timestamp; opcjonalny `delay_ms` |
| `POST` | `/sessions/{id}/capture/images` | Upload zdjęcia pomiarowego (multipart) |

### Pomiar
| Metoda | Endpoint | Opis |
|--------|----------|------|
| `POST` | `/sessions/{id}/measure` | Uruchom pipeline 3D w tle |
| `GET` | `/sessions/{id}/measurement` | Wyniki: W/L/H mm, 3× objętość, walidacja |
| `GET` | `/sessions/{id}/measurement/report` | Pełny raport tekstowy |
| `POST` | `/measure/synthetic` | Test bez kamer (dane syntetyczne) |

### Inne
| Metoda | Endpoint | Opis |
|--------|----------|------|
| `GET` | `/health` | Health check |
| `WS` | `/ws/{session_id}/{device_id}` | WebSocket — eventy, sync, heartbeat |

---

## Testy

```bash
pytest test_calibration.py   # Zhang, stereo, serializacja, wysokie rozdzielczości
pytest test_measurement.py   # RANSAC, bbox, objętość, walidacja
pytest                        # wszystkie testy
```

Testy działają w pełni syntetycznie — nie wymagają kamer ani plików obrazów.

---

## Zmienne środowiskowe

| Zmienna | Default (config.py) | Opis |
|---------|---------------------|------|
| `CHECKERBOARD_ROWS` | `5` | Wewnętrzne narożniki szachownicy (wiersze) |
| `CHECKERBOARD_COLS` | `8` | Wewnętrzne narożniki szachownicy (kolumny) |
| `SQUARE_SIZE_MM` | `15.0` | Rozmiar kwadratu [mm] |
| `CALIBRATION_DIR` | `/app/data/calib` | Katalog ze zdjęciami szachownicy |
| `CALIBRATION_OUTPUT` | `/app/data/calib_output` | Katalog wyników kalibracji |
| `BACKEND_PORT` | `8000` | Port serwera |
| `LOG_LEVEL` | `INFO` | Poziom logów konsolowych |
| `CORNER_DETECT_MAX_WIDTH` | `1920` | Skalowanie dla kamer hi-res (px) |
| `MIN_CALIBRATION_IMAGES` | `3` | Minimalna liczba klatek do kalibracji |
| `MAX_STEREO_REPROJ_ERROR` | `2.0` | Próg RMS (px) — powyżej ostrzeżenie |

Stałe hardcoded w `config.py` (nieeksponowane przez `.env`):
- `PALLET_HEIGHT_MM = 144.0` — standardowa wysokość europalety

---

## Uwagi implementacyjne

**Brak Redis i Celery.** Sesje trzymane w pamięci z persystencją JSON na dysk. Zadania CPU-intensive przez `asyncio.to_thread()`. Wystarczające dla 2 urządzeń i jednej sesji pomiarowej na raz.

**Rozłączenie = rekalibracja.** Brak mechanizmu reconnect. Zerwanie WebSocket przez dowolne urządzenie wymaga ponownej kalibracji stereo.

**Identyfikacja pary stereo przez MAC.** Parametry kalibracji zapisywane w `data/{session_id}/stereo.json` są powiązane z MAC adresami obu urządzeń.

**Skalowanie hi-res.** Detekcja narożników szachownicy na przeskalowanym obrazie (`CORNER_DETECT_MAX_WIDTH`), refinement na pełnej rozdzielczości — dla kamer 12+ Mpx w telefonach.

**CORS.** Aktualnie `allow_origins=["*"]` — do ograniczenia przed wdrożeniem produkcyjnym.
