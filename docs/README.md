# Dokumentacja — Stereo Vision (SWIZ_CV)

System stereowizyjny do pomiaru wymiarów obiektów na europalecie (1200×800 mm).
Dwie kamery mobilne → FastAPI → pipeline OpenCV → wymiary w milimetrach.

---

## Mapa dokumentów

| Plik | Co opisuje |
|------|-----------|
| [architecture.md](architecture.md) | Diagram systemu, maszyna stanów sesji, przepływ danych |
| [backend.md](backend.md) | Serwer FastAPI — sesje, WebSocket, zadania w tle |
| [api-reference.md](api-reference.md) | Wszystkie endpointy REST + protokół WebSocket |
| [pipeline-cv.md](pipeline-cv.md) | 7-etapowy pipeline wizji komputerowej |
| [calibration.md](calibration.md) | Kalibracja stereo metodą Zhanga — krok po kroku |
| [flutter.md](flutter.md) | Aplikacja mobilna Flutter — ekrany, modele, serwisy |
| [deployment.md](deployment.md) | Uruchomienie Docker, lokalnie, testy |

---

## Szybki start

```bash
# Backend (Docker)
docker-compose up

# Testy lokalne (bez kamer)
python pipeline.py

# Testy jednostkowe
python -m pytest test_calibration.py test_measurement.py -v

# Flutter
cd flutter_app && flutter run -d chrome
```

---

## Struktura repozytorium

```
SWIZ_CV/
├── backend/             # FastAPI serwer
│   ├── main.py          # wszystkie endpointy HTTP + WebSocket
│   ├── session.py       # maszyna stanów sesji + SessionStore
│   ├── tasks.py         # zadania w tle (kalibracja, pomiar) + WSManager
│   └── schemas.py       # modele Pydantic (request/response)
│
├── calibration.py       # metoda Zhanga — kalibracja mono i stereo
├── disparity.py         # rektyfikacja, SGBM, konwersja dysparycja→głębokość
├── pointcloud.py        # budowanie chmury punktów z dysparycji
├── pallet.py            # wykrywanie płaszczyzny palety (RANSAC + SVD)
├── measurement.py       # segmentacja obiektu, bounding box, walidacja
├── pipeline.py          # pełny pipeline 7-etapowy + tryb syntetyczny
├── config.py            # konfiguracja z .env
│
├── flutter_app/         # aplikacja mobilna (poza Dockerem)
│   └── lib/
│       ├── main.dart
│       ├── models/
│       ├── providers/
│       ├── screens/
│       └── services/
│
├── test_calibration.py  # testy kalibracji (dane syntetyczne)
├── test_measurement.py  # testy pallet + pomiaru (dane syntetyczne)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```
