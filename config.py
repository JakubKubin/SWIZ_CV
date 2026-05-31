# config.py
"""Centralna konfiguracja systemu stereo vision.

Wartosci zalezne od srodowiska uruchomieniowego (sciezki Docker vs lokalne,
port) sa czytane ze zmiennych srodowiskowych (.env). Pozostale parametry to
stale strojeniowe zwiazane z fizycznym ukladem (szachownica, geometria,
progi pomiaru) - trzymane bezposrednio tutaj, bo nie zmieniaja sie miedzy
srodowiskami, a jedno miejsce ulatwia ich utrzymanie.
"""
import os
import cv2
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Szachownica kalibracyjna
# ---------------------------------------------------------------------------
# ROWS x COLS to liczba wewnetrznych naroznikow.
BOARD_ROWS: int = 6
BOARD_COLS: int = 8
SQUARE_SIZE_MM: float = 42.0   # fizyczny rozmiar kwadratu [mm]

# Automatyczny obrot wszystkich przesylanych zdjec (0, 90, 180, 270).
# Przydatne, gdy telefony sa fizycznie w poziomie (landscape), ale zapisuja
# pliki w pionie (portrait).
IMAGE_ROTATE: int = 270

# ---------------------------------------------------------------------------
# Sciezki (zalezne od srodowiska: Docker /app/data vs lokalne ./calib_images)
# ---------------------------------------------------------------------------
CALIB_DIR: str = os.environ.get("CALIBRATION_DIR", "./calib_images")
CALIB_OUTPUT: str = os.environ.get("CALIBRATION_OUTPUT", "./calib_output")

# ---------------------------------------------------------------------------
# Europaleta (wg normy EUR/EPAL 1)
# ---------------------------------------------------------------------------
# Wymiary standardowej europalety uzywane do detekcji plaszczyzny (RANSAC)
# i filtrowania ROI — punkty spoza obrysu palety sa odrzucane.
PALLET_WIDTH_MM: float = 360.0
PALLET_LENGTH_MM: float = 290.0
PALLET_HEIGHT_MM: float = 112.0   # uwaga: stala dokumentacyjna, obecnie nieuzywana w kodzie

# ---------------------------------------------------------------------------
# Progi jakosci kalibracji
# ---------------------------------------------------------------------------
MIN_CALIBRATION_IMAGES: int = 3        # minimalna liczba par obrazow z wykrytym wzorcem
MAX_SINGLE_REPROJ_ERROR: float = 1.0   # [px] prog ostrzezenia dla kalibracji pojedynczej kamery
MAX_STEREO_REPROJ_ERROR: float = 2.0   # [px] prog akceptacji dla kalibracji stereo (wyzszy - bo 2 kamery)

# ---------------------------------------------------------------------------
# Parametry detekcji naroznikow szachownicy
# ---------------------------------------------------------------------------
# Detekcja naroznikow na pelnej rozdzielczosci telefonow (3-4K) jest wolna bez
# poprawy jakosci. Obrazy szersze niz ta wartosc sa tymczasowo zmniejszane
# WYLACZNIE na czas detekcji - wykryte narozniki sa nastepnie przeskalowane
# z powrotem do wspolrzednych oryginalu i dorefinowane subpikselowo. Dzieki
# temu kalibracja zawsze odbywa sie w natywnej rozdzielczosci obrazow.
CORNER_DETECT_MAX_WIDTH: int = 4080
# Okno 11x11 px
CORNER_SUBPIX_WIN: tuple[int, int] = (11, 11)
# (-1, -1) oznacza brak strefy martwej wokol srodka okna
CORNER_SUBPIX_ZERO_ZONE: tuple[int, int] = (-1, -1)
# Kryterium stopu dla cornerSubPix i stereoCalibrate:
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ---------------------------------------------------------------------------
# Pipeline pomiarowy
# ---------------------------------------------------------------------------
MAX_DEPTH_MM: float = 1500.0          # piksele glebsze niz ten prog sa odrzucane
MIN_DEPTH_MM: float = 700.0           # najblizsza spodziewana odleglosc obiektu [mm]
NOISE_FLOOR_MM: float = 10.0          # min. wysokosc nad paleta uznawana za obiekt [mm]
MAX_OBJECT_HEIGHT_MM: float = 200.0   # max. wysokosc obiektu (filtr "latajacych" punktow) [mm]
SGBM_NUM_DISPARITIES: int = 1600      # musi byc wielokrotnoscia 16

# ---------------------------------------------------------------------------
# Obslugiwane formaty obrazow (kolejnosc wyszukiwania)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: list[str] = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
