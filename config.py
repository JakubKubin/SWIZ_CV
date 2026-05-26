# config.py
"""Centralna konfiguracja systemu stereo vision.

Wszystkie parametry sa czytane ze zmiennych srodowiskowych (plik .env),
co pozwala dostosowac system do roznych szachownic i srodowisk bez
modyfikacji kodu.
"""
import os
import cv2
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Szachownica kalibracyjna
# ---------------------------------------------------------------------------
# ROWS x COLS to liczba wewnetrznych naroznikow.

BOARD_ROWS: int = int(os.environ.get("CHECKERBOARD_ROWS", 5))
BOARD_COLS: int = int(os.environ.get("CHECKERBOARD_COLS", 8))
SQUARE_SIZE_MM: float = float(os.environ.get("SQUARE_SIZE_MM", 44.0))  # fizyczny rozmiar kwadratu [mm]

# Automatyczny obrót wszystkich przesyłanych zdjęć (0, 90, 180, 270)
# Przydatne, gdy telefony są fizycznie w poziomie (landscape), ale zapisują pliki w pionie (portrait).
IMAGE_ROTATE: int = int(os.environ.get("IMAGE_ROTATE", 0))



# ---------------------------------------------------------------------------
# Sciezki
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
PALLET_HEIGHT_MM: float = 112.0

# ---------------------------------------------------------------------------
# Progi jakosci kalibracji
# ---------------------------------------------------------------------------
MIN_CALIBRATION_IMAGES: int = 3        # minimalna liczba par obrazow z wykrytym wzorcem
MAX_SINGLE_REPROJ_ERROR: float = 1.0   # [px] prog akceptacji dla kalibracji pojedynczej kamery
MAX_STEREO_REPROJ_ERROR: float = 2.0   # [px] prog akceptacji dla kalibracji stereo (wyzszy - bo 2 kamery)

# ---------------------------------------------------------------------------
# Parametry detekcji naroznikow szachownicy
# ---------------------------------------------------------------------------
# Detekcja naroznikow na pelnej rozdzielczosci telefonow (3-4K) jest wolna bez
# poprawy jakosci. Obrazy szersze niz ta wartosc sa tymczasowo zmniejszane
# WYLACZNIE na czas detekcji - wykryte narozniki sa nastepnie przeskalowane
# z powrotem do wspolrzednych oryginalu i dorefinowane subpikselowo. Dzieki
# temu kalibracja zawsze odbywa sie w natywnej rozdzielczosci obrazow.
CORNER_DETECT_MAX_WIDTH: int = int(os.environ.get("CORNER_DETECT_MAX_WIDTH", 1920))
# Okno 11x11 px
CORNER_SUBPIX_WIN: tuple[int, int] = (11, 11)
# (-1, -1) oznacza brak strefy martwej wokol srodka okna
CORNER_SUBPIX_ZERO_ZONE: tuple[int, int] = (-1, -1)

# Kryterium stopu dla cornerSubPix i stereoCalibrate:

TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ---------------------------------------------------------------------------
# Pipeline pomiarowy
# ---------------------------------------------------------------------------
MAX_DEPTH_MM: float = float(os.environ.get("MAX_DEPTH_MM", 5000.0))
MIN_DEPTH_MM: float = float(os.environ.get("MIN_DEPTH_MM", 800.0))
NOISE_FLOOR_MM: float = float(os.environ.get("NOISE_FLOOR_MM", 20.0))
SGBM_NUM_DISPARITIES: int = int(os.environ.get("SGBM_NUM_DISPARITIES", 128))  # musi byc wielokrotnoscia 16

# ---------------------------------------------------------------------------
# Obsługiwane formaty obrazow (kolejnosc wyszukiwania)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: list[str] = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]