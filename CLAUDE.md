# Stereo Vision - Pomiar obiektow na europalecie

## Cel projektu
System stereowizyjny do pomiaru wymiarow obiektow umieszczonych na europalecie
(1200x800mm) z uzyciem wielu kamer.

## Pipeline - Etapy

### ETAP 1: Kalibracja kamer (CURRENT)
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
- SGBM / BM matching
- Filtracja mapy glebi (WLS filter)
- Konwersja disparity -> depth (mm)

### ETAP 5: Detekcja europalety
- Wykrycie plaszczyzny palety w chmurze punktow
- Definicja ROI na podstawie wymiarow palety (1200x800mm)
- Transformacja do ukladu wspolrzednych palety

### ETAP 6: Segmentacja obiektu
- Oddzielenie obiektu od palety (depth thresholding)
- Kontur obiektu w 3D
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
