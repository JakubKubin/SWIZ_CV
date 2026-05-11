# Wdrożenie i uruchomienie

---

## Konfiguracja .env

Przed uruchomieniem skonfiguruj `.env` w katalogu głównym projektu:

```env
# Szachownica kalibracyjna
CHECKERBOARD_ROWS=9       # liczba wewnętrznych narożników — wiersze
CHECKERBOARD_COLS=6       # liczba wewnętrznych narożników — kolumny
SQUARE_SIZE_MM=25.0       # rozmiar kwadratu [mm]

# Ścieżki (używane przy uruchomieniu lokalnym bez Dockera)
CALIBRATION_DIR=/app/data/calib
CALIBRATION_OUTPUT=/app/data/calib_output

# Port serwera
BACKEND_PORT=8000
```

Przykładowy plik z wartościami testowymi: `env.example`.

---

## Uruchomienie — Docker (zalecane)

```bash
# Zbuduj i uruchom backend
docker-compose up

# W tle
docker-compose up -d

# Zatrzymaj i usuń kontenery
docker-compose down

# Usuń też wolumen z danymi sesji
docker-compose down -v
```

Serwis `backend` startuje automatycznie z health checkiem:
- URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Volume: `session_data:/app/data` (dane sesji i kalibracji)

### docker-compose.yml

```yaml
services:
  backend:
    build: .
    ports:
      - "${BACKEND_PORT:-8000}:8000"
    volumes:
      - session_data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  session_data:
```

---

## Uruchomienie lokalne (bez Dockera)

### Wymagania

- Python 3.12+
- Zależności systemowe OpenCV: `libgl1`, `libglib2.0-0` (Linux) lub odpowiedniki

```bash
# Utwórz środowisko wirtualne
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom serwer
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Testy

### Testy jednostkowe (bez kamer)

```bash
# Wszystkie testy
python -m pytest test_calibration.py test_measurement.py -v

# Z wizualizacją (zapisuje obrazy do test_output/)
python -m pytest test_calibration.py -v --visualize

# Tylko konkretna klasa
python -m pytest test_calibration.py::TestStereoCalibration -v
```

Co jest testowane:
- `test_calibration.py` — kalibracja mono i stereo na danych syntetycznych (warpAffine + projectPoints)
- `test_measurement.py` — RANSAC płaszczyzny palety, transformacja układu, segmentacja, bounding box, walidacja

### Tryb syntetyczny (pipeline bez kamer)

```bash
# Uruchamia pełny 7-etapowy pipeline na scenie wirtualnej
python pipeline.py

# Tryb realny (wymaga kalibracji i zdjęć)
python pipeline.py --mode real \
  --calib data/stereo.json \
  --left captures/left.jpg \
  --right captures/right.jpg
```

Wyniki zapisywane do `pipeline_output/`.

### Test API bez kamer

```bash
# Przez curl
curl -X POST http://localhost:8000/measure/synthetic

# Przez Swagger UI
# → http://localhost:8000/docs → POST /measure/synthetic → Execute
```

---

## Uruchomienie Flutter

```bash
cd flutter_app
flutter pub get

# Web (testy UI)
flutter run -d chrome

# Android (podłącz telefon USB z włączonym debugowaniem)
flutter run -d android

# iOS (wymaga macOS + Xcode)
flutter run -d ios

# Build APK release
flutter build apk --release
```

Na telefonie fizycznym zmień URL serwera w aplikacji na adres IP komputera w sieci lokalnej:
`http://192.168.1.X:8000`

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

# Zależności systemowe dla OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Moduły CV i backend
COPY calibration.py disparity.py pallet.py measurement.py \
     pipeline.py pointcloud.py config.py ./
COPY backend/ backend/

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Struktura danych na dysku

Po uruchomieniu sesji backend tworzy katalogi w `/app/data/` (lub `./data/` lokalnie):

```
data/
└── {session_id}/              # 8-znakowy hex, np. a3f9b12c
    ├── calib/
    │   ├── {device_id}/       # lewa kamera
    │   │   ├── frame_0000.jpg
    │   │   ├── frame_0001.jpg
    │   │   └── ...
    │   └── {device_id}/       # prawa kamera
    │       └── ...
    ├── captures/
    │   ├── {device_id}/       # lewa kamera — zdjęcia pomiarowe
    │   │   └── capture_0000.jpg
    │   └── {device_id}/       # prawa kamera
    │       └── capture_0000.jpg
    ├── stereo.json             # parametry kalibracji (po compute)
    ├── cloud.ply               # chmura punktów (po pomiarze)
    └── measurement_report.txt  # raport tekstowy (po pomiarze)
```

Usunięcie sesji (`DELETE /sessions/{id}`) usuwa katalog `data/{session_id}/` z dysku.

---

## Wymagania sprzętowe

| Komponent | Minimum |
|-----------|---------|
| RAM | 2 GB (pipeline OpenCV) |
| CPU | 2 rdzenie (operacje CV są CPU-bound) |
| Dysk | 1 GB wolnego miejsca (na dane sesji) |
| Sieć | WiFi 5 GHz zalecane (upload zdjęć RAW) |

Telefony na statywie, odległość 50–100 cm od obiektu, baza stereo ~10–20 cm.
