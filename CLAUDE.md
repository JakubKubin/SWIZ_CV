# Stereo Vision - Pomiar obiektow na europalecie

## Cel projektu
System stereowizyjny do pomiaru wymiarów (długość, szerokość, wysokość) obiektów umieszczonych na standardowej europalecie (1200x800 mm).
Wykorzystuje on technikę stereowizji realizowaną przez grupę urządzeń mobilnych/webowych działających w ramach jednej sesji pomiarowej.


W skrócie:
Akwizycja zdjęć ma się odbywać przy pomocy aplikacji mobilnej lub webowej napisanej we flutterze.
Docker compose jako orkiestrator w którym będzie flutter i python jako backend jako fastapi do komunikacji.
W aplikacji mobilnej powinna istnieć sesja do której mogą dołączać się urządzenia. Urządzenia po zamknięciu sesji na nowe powinny być skalibrowane przy pomocy metody Zhanga.
Po kalibracji przystępujemy do wykonania zdjęcia, urządzenie pierwsze w sesji zarządza wykonaniem zdjęcia symulatanicznym na tym urządzeniu i reszcie urządzeń w sesji.
Akwizycja zdjęcia w tym samym momencie z wielu kamer na raz ma pozwolić na wyznaczenie mapy głębi i chumury punktów w przestrzeni.
Frontend powinien zostać wykonany w frameworku Flutter w języku Dart. Komunikacja z backendem poprzez API i WebSockets - Python fastapi.


## Architektura systemu

1. Frontend - Flutter, aplikacja mobilna odpowiedzialna za zarządzanie sesją, podgląd z kamery akwizycję zdjęć oraz wyświetlanie wyników
2. Backend - Serwer obliczeniowy realizujący kalibrację, rektyfikację, generowanie mapy głębi i analizę chmury punktów.
3. Komunikacja:
	- WebSockets (do zarządzania sesją i synchronicznego wyzwalania zdjęć)
	- REST API dla przesyłu danych
4. Baza danych/Storage: Redis (do szybkiego przekazywania statusu sesji) + wolumeny Dockerowe na zdjęcia.

Architektura Wdrożeniowa
Komponent 		Technologia 		Rola
Orkiestracja	Docker Compose		Zarządzanie kontenerami
API				FastAPI + Uvicorn	Komunikacja asynchroniczna
Worker			Celery + Redis		Ciężkie obliczenia OpenCV/Open3D
Storage			Shared Volume		Przechowywanie zdjęć RAW i wyników
Frontend		Flutter				Interfejs użytkownika i dostęp do kamery (Camera2 API / AVFoundation)

Flutter znajduje się w docker compose ze względu na przygotowanie aplikacji webowej i aplikacji mobilnej na raz po to aby przetestować podstawowe funkcje + UI/UX.
Faktycznie aplikacja będzie ostatecznie na Androida i iOS. Nie wrzucamy fluttera do Dockera.

Zakładamy na start dwa urządzenia ale piszemy aplikację i komunikację tak jakby mogłoby być więcej.
Zdjęcia będą realizowane telefonami na statywach.
Załóżmy że zdjęcia będą realizowane w odległości 50-100cm od odbiektu
Zapisujemy dane z kalibracji danych pary urządzeń poprzez ich MAC - wiemy dokładnie jak zidentyfikować
Dodajemy możliwość wyczyszczenia danych dla użytkownika jego par
Jeśli urządzenie się jakieś rozłączy to przykro mi trzeba rekonfigurować i jazda spowrotem

## Logika biznesowa

3.1. Zarządzanie Sesją
Tworzenie sesji: Użytkownik (Lider) tworzy nową sesję. Serwer generuje unikalny identyfikator sesji.
Dołączanie: Pozostałe urządzenia (Followers) dołączają do sesji. Każde urządzenie jest identyfikowane przez device_id.
Role: Lider posiada uprawnienia do wyzwalania kalibracji i wykonania zdjęcia pomiarowego.

3.2. Proces Kalibracji
Przed pomiarem system wymaga kalibracji par stereo.
Urządzenia muszą pozostać nieruchome względem siebie od momentu kalibracji do wykonania zdjęcia.
Wykorzystywana jest metoda Zhanga z użyciem wzorca szachownicy.

3.3. Akwizycja i Synchronizacja
Wyzwalanie zdjęcia odbywa się symultanicznie. Backend wysyła sygnał przez WebSockets do wszystkich urządzeń w sesji.
Urządzenia wykonują zdjęcie i przesyłają je na serwer wraz z metadanymi (identyfikator urządzenia, timestamp).

4. Endpointy API (FastAPI)
Obsługa Sesji
Metoda	Endpoint	Opis
POST	/sessions/create 		Tworzy nową sesję pomiarową.
POST	/sessions/{id}/join 	Dołączenie urządzenia do sesji.
GET 	/sessions/{id}/status	Zwraca listę połączonych urządzeń.


Kalibracja i Akwizycja
Metoda	Endpoint	Opis
POST	/calibration/upload		Przesłanie zdjęć szachownicy dla danego urządzenia.
POST	/calibration/compute	Uruchomienie obliczeń parametrów wewnętrznych i zewnętrznych.
POST	/capture/trigger		Wysłanie sygnału do wszystkich urządzeń o wykonaniu zdjęcia.
POST	/capture/upload			Przesłanie finalnego zdjęcia obiektów do przetworzenia.


POST /sessions/{id}/join
Body: {"device_id": "UUID", "model": "iPhone13", "is_leader": bool}

Logic: Rejestruje urządzenie w Redis. Jeśli to Lider, inicjuje sesję.

POST /calibration/upload
Body: multipart/form-data (file, device_id, frame_id)

Logic: Zapisuje zdjęcie w folderze /data/{session_id}/{device_id}/calib/.

POST /calibration/compute
Logic: 1. Pobiera obrazy z dysku.
2. Uruchamia cv2.stereoCalibrate.
3. Generuje mapy rektyfikacji (initUndistortRectifyMap).
4. Zwraca RMS error. Jeśli > 0.5, sugeruje powtórzenie kalibracji.

POST /capture/upload
Body: multipart/form-data (file, timestamp, device_id)

Logic: To jest "zdjęcie właściwe". Po odebraniu kompletu zdjęć od wszystkich device_id w sesji, serwer automatycznie odpala worker (Celery/BackgroundTasks) do obliczeń 3D.

Wyniki
Metoda	Endpoint	Opis
GET	/measurement/{session_id}	Pobranie wyników pomiaru (wymiary w mm).

Zarządzanie sesją musi obsługiwać dynamiczne dołączanie urządzeń i utrzymywanie ich stanu (State Machine).

Stany Sesji:
IDLE: Oczekiwanie na urządzenia.

CALIBRATING: Zbieranie par zdjęć szachownicy.

READY: Parametry stereo wyliczone, system gotowy do pomiaru.

PROCESSING: Serwer przetwarza chmurę punktów.

Sekwencja zdarzeń (Workflow):
Handshake: Urządzenie wysyła POST /sessions/{id}/join z informacjami o specyfikacji aparatu (fov, resolution). Serwer przypisuje device_index (np. 0 dla Lidera, 1 dla pierwszego Followera).

Utrzymanie połączenia: Każde urządzenie otwiera WebSocket /ws/{session_id}/{device_id}. Służy on do przesyłania komend typu START_CALIBRATION, TRIGGER_CAPTURE oraz sygnałów HEARTBEAT.

Wybór Pary: W sesji wielourządzeniowej Backend musi wiedzieć, które urządzenia tworzą parę stereo (np. Device A i Device B patrzą na ten sam obiekt).

Protokół Synchronicznej Akwizycji (Precision Sync)
Problem opóźnienia sieciowego (Network Jitter) rozwiązujemy poprzez Buffered Delayed Capture:

Synchronizacja Czasu: Przy połączeniu WebSocket, klient i serwer wykonują uproszczony protokół NTP, aby wyliczyć offset czasu lokalnego względem serwera.

Komenda Trigger: Lider klika przycisk. Serwer oblicza Target_Timestamp = Server_Now + 1000ms.

Broadcast: Serwer wysyła do wszystkich: {"action": "capture", "at": Target_Timestamp}.

Hardware Capture: Urządzenia (Flutter) planują wykonanie zdjęcia dokładnie na Target_Timestamp (używając czasu skorygowanego o offset). Dzięki temu błąd synchronizacji spada.

Architektura folderów:
stereo-vision/
├── docker-compose.yml
├── .env
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # ustawienia z .env
│   │   ├── models/
│   │   │   ├── session.py       # State machine sesji
│   │   │   └── device.py        # Model urządzenia
│   │   ├── routers/
│   │   │   ├── sessions.py      # /sessions/*
│   │   │   ├── calibration.py   # /calibration/*
│   │   │   ├── capture.py       # /capture/*
│   │   │   └── measurement.py   # /measurement/*
│   │   ├── websocket/
│   │   │   └── manager.py       # WebSocket connection manager
│   │   ├── services/
│   │   │   ├── calibration_svc.py   # Logika Zhanga, zapis JSON per MAC-para
│   │   │   ├── stereo_svc.py        # Rektyfikacja, disparity, głębia
│   │   │   ├── pointcloud_svc.py    # Open3D, RANSAC, bounding box
│   │   │   └── storage_svc.py       # Zapis/odczyt plików
│   │   └── workers/
│   │       └── tasks.py         # Celery tasks (ciężkie obliczenia)
│   └── data/                    # Docker volume
│       ├── sessions/
│       └── calibrations/        # params_{macA}_{macB}.json
│
└── flutter_app/                 # Osobno, poza Dockerem
    └── ...
	
## Pipeline - Etapy

### ETAP 1: Kalibracja kamer
- Kalibracja pojedynczej kamery (metoda Zhanga, OpenCV)
- Kalibracja stereo (para kamer)
- Zapis/odczyt parametrow kalibracji (JSON)
- Testy jednostkowe i walidacja reproj. error
- Zmienne srodowiskowe: rozmiar szachownicy, rozmiar kwadratu

### ETAP 2: Akwizycja obrazow
- Jednoczesne przechwytywanie z wielu kamer
- Synchronizacja klatek
- Zapis par stereo do dalszej obrobki

### ETAP 3: Rektyfikacja stereo
- Obliczenie macierzy rektyfikacji z parametrow kalibracji
- Remapowanie obrazow do ukladu rownoleglego
- Walidacja rektyfikacji (linie epipolarne)

### ETAP 4: Mapa glebi (disparity map)
- SGBM (Semi-Global Block Matching)
- Filtracja mapy glebi (WLS filter)
- Konwersja disparity -> depth (mm)

### ETAP 5: Detekcja europalety
- Wykrycie plaszczyzny palety w chmurze punktow RANSAC na chmurze punktów (płaszczyzna dominująca = podłoga palety)
- Definicja ROI na podstawie wymiarow palety (1200x800mm)
- Transformacja do ukladu wspolrzednych palety

### ETAP 6: Segmentacja obiektu
- Oddzielenie obiektu od palety - usunięcie punktów należących do płaszczyzny palety (clipping)
- Kontur obiektu w  - pozostałe punkty powyżej pewnego progu (noise floor) są traktowane jako obiekt.
- Bounding box 3D

### ETAP 7: Pomiar wymiarow
- Szerokosc, dlugosc, wysokosc obiektu (mm)
- Walidacja wzgledem znanych wymiarow palety
- Raport pomiarowy

## Zmienne srodowiskowe (.env)
- CHECKERBOARD_ROWS - liczba wewnetrznych naroznikow (wiersze)
- CHECKERBOARD_COLS - liczba wewnetrznych naroznikow (kolumny)
- SQUARE_SIZE_MM - rozmiar kwadratu szachownicy w mm
- CALIBRATION_DIR - sciezka do obrazow kalibracyjnych
- CALIBRATION_OUTPUT - sciezka do zapisu parametrow

.env.example:
CHECKERBOARD_ROWS=9
CHECKERBOARD_COLS=6
SQUARE_SIZE_MM=25
CALIBRATION_DIR=/app/data/calib
CALIBRATION_OUTPUT=/app/data/params.json



Wdrożenie:
FAZA 1 — Fundament backendu
├── docker-compose + .env + config.py
├── State machine sesji (Redis)
├── WebSocket manager
└── Endpointy /sessions/*

FAZA 2 — Kalibracja
├── Upload zdjęć szachownicy
├── Obliczenia Zhang (OpenCV)
└── Zapis/odczyt per para MAC

FAZA 3 — Akwizycja
├── Precision Sync (CAPTURE_DELAY_MS)
├── Upload zdjęć właściwych
└── Trigger Celery worker

FAZA 4 — Pipeline 3D
├── Rektyfikacja stereo
├── Mapa głębi (SGBM + WLS)
├── Chmura punktów (Open3D)
├── RANSAC (detekcja palety)
└── Bounding box - wymiary

Flutter
├── Zarządzanie sesją + WS
├── Kamera + sync capture
└── Wyświetlanie wyników
