"""Konfiguracja projektu stereo vision - pomiar obiektow na europalecie.

Wartosci domyslne moga byc nadpisane przez zmienne srodowiskowe (patrz env.example).
"""
import os
import cv2

# ---------------------------------------------------------------------------
# Szachownica kalibracyjna
# ---------------------------------------------------------------------------
BOARD_ROWS: int = int(os.environ.get("CHECKERBOARD_ROWS", 9))   # wewnetrzne narozniki (wiersze)
BOARD_COLS: int = int(os.environ.get("CHECKERBOARD_COLS", 6))   # wewnetrzne narozniki (kolumny)
SQUARE_SIZE_MM: float = float(os.environ.get("SQUARE_SIZE_MM", 25.0))  # rozmiar kwadratu [mm]

# ---------------------------------------------------------------------------
# Sciezki
# ---------------------------------------------------------------------------
CALIB_DIR: str = os.environ.get("CALIBRATION_DIR", "./calib_images")
CALIB_OUTPUT: str = os.environ.get("CALIBRATION_OUTPUT", "./calib_output")

# ---------------------------------------------------------------------------
# Europaleta (wg normy EUR/EPAL)
# ---------------------------------------------------------------------------
PALLET_WIDTH_MM: float = 1200.0
PALLET_LENGTH_MM: float = 800.0
PALLET_HEIGHT_MM: float = 144.0  # standardowa wysokosc palety

# ---------------------------------------------------------------------------
# Progi jakosci kalibracji
# ---------------------------------------------------------------------------
MIN_CALIBRATION_IMAGES: int = 3        # minimalna liczba obrazow do kalibracji
MAX_SINGLE_REPROJ_ERROR: float = 1.0   # [px] max akceptowalny blad reproj. dla pojedynczej kamery
MAX_STEREO_REPROJ_ERROR: float = 2.0   # [px] max akceptowalny blad reproj. dla stereo

# ---------------------------------------------------------------------------
# Parametry detekcji naroznikow szachownicy
# ---------------------------------------------------------------------------
CORNER_SUBPIX_WIN: tuple[int, int] = (11, 11)   # okno wyszukiwania subpix
CORNER_SUBPIX_ZERO_ZONE: tuple[int, int] = (-1, -1)

# Kryterium stopu dla cornerSubPix i stereoCalibrate
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ---------------------------------------------------------------------------
# Obsługiwane formaty obrazow (kolejnosc wyszukiwania)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: list[str] = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
