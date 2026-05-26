# calibration.py
"""Modul kalibracji kamer - metoda Zhanga (OpenCV).

Obsluguje kalibracje pojedynczej kamery oraz kalibracje stereo.
Wyniki kalibracji (macierze K, wspolczynniki dystorsji, R, T, Q)
sa zapisywane do pliku JSON i wczytywane przez pozostale moduly
(disparity.py, pipeline.py, backend/tasks.py).


Przykladowe uzycie CLI:
    python calibration.py --mode stereo \\
        --left-dir calib_images/left \\
        --right-dir calib_images/right \\
        --output calib_output/stereo.json
"""
import os, json, glob, logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, overload, Literal
import numpy as np
import cv2

import config
from logging_setup import setup_logging

setup_logging()
log = logging.getLogger(__name__)

# Lokalne aliasy stalych z config - skracaja zapis w calym module
BOARD_ROWS = config.BOARD_ROWS
BOARD_COLS = config.BOARD_COLS
SQUARE_SIZE = config.SQUARE_SIZE_MM
CALIB_DIR = config.CALIB_DIR
CALIB_OUT = config.CALIB_OUTPUT
CRITERIA = config.TERM_CRITERIA


# ---------------------------------------------------------------------------
# Struktury danych dla wykrytych punktow kalibracyjnych
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Zebrane punkty kalibracyjne dla pojedynczej kamery.

    obj_points - wspolrzedne 3D naroznikow szachownicy w ukladzie wzorca
                 (Z=0, bo szachownica jest plaska). Jednostka: mm.
    img_points - odpowiadajace im wspolrzedne 2D na obrazie [px].
    Obie listy musza miec te sama dlugosc - kazdy element to jedna klatka.
    """
    obj_points: list   # list[np.ndarray shape (N,3)] - wspolrzedne 3D wzorca
    img_points: list   # list[np.ndarray shape (N,1,2)] - punkty 2D na obrazie
    image_size: tuple[int, int]  # (width, height) - potrzebne do calibrateCamera

    def __len__(self) -> int:
        return len(self.obj_points)


@dataclass
class StereoCalibrationData:
    """Zebrane punkty kalibracyjne dla pary stereo.

    Zawiera wylacznie pary klatek, gdzie OBIE kamery wykryly wzorzec.
    Pary niekompletne (tylko jedna kamera widzi szachownice) sa odrzucane,
    bo stereoCalibrate wymaga odpowiadajacych sobie punktow z obu kamer.
    """
    obj_points: list        # wspolrzedne 3D - wspolne dla obu kamer
    left_points: list       # punkty 2D z lewej kamery
    right_points: list      # punkty 2D z prawej kamery
    image_size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.obj_points)

    @property
    def left(self) -> CalibrationData:
        """Dane lewej kamery sformatowane jako CalibrationData.
        Pozwala przekazac je bezposrednio do _calibrate_from_data()."""
        return CalibrationData(self.obj_points, self.left_points, self.image_size)

    @property
    def right(self) -> CalibrationData:
        """Dane prawej kamery sformatowane jako CalibrationData.
        Pozwala przekazac je bezposrednio do _calibrate_from_data()."""
        return CalibrationData(self.obj_points, self.right_points, self.image_size)


# ---------------------------------------------------------------------------
# Parametry kamer
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    """Parametry wewnetrzne pojedynczej kamery.

    camera_matrix - macierz wewnetrzna K (3x3): ogniskowe fx, fy i punkt glowny cx, cy
    dist_coeffs   - wspolczynniki dystorsji obiektywu (k1,k2,p1,p2,k3)
    reproj_error  - RMS bledu reprojekcji z kalibracji [px]; im nizszy, tym lepsza kalibracja
    image_size    - (width, height) obrazow uzytych do kalibracji
    """
    camera_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    reproj_error: float = 0.0
    image_size: tuple[int, int] = (0, 0)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Zwraca obraz skorygowany o dystorsje obiektywu.
        Przydatne do podgladu - w pipeline uzywamy remap() dla wydajnosci."""
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def to_dict(self):
        """Serializuje parametry do slownika (do zapisu JSON)."""
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reproj_error": self.reproj_error,
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, d):
        """Odtwarza obiekt z slownika (wczytanego z JSON)."""
        return cls(
            camera_matrix=np.array(d["camera_matrix"]),
            dist_coeffs=np.array(d["dist_coeffs"]),
            reproj_error=d["reproj_error"],
            image_size=tuple(d["image_size"]),
        )


@dataclass
class StereoParams:
    """Pelne parametry systemu stereo - kalibracja obu kamer + ich wzajemne polozenie.

    Pola geometrii stereo:
      R, T - rotacja i translacja z lewej do prawej kamery (wynik stereoCalibrate)
      E    - macierz esencjalna (zawiera R i T w skondensowanej formie)
      F    - macierz fundamentalna (jak E, ale w pikselach; przydatna do rysowania linii epipolarnych)

    Pola rektyfikacji (wynik stereoRectify, uzywane przez rectify_maps()):
      R1, R2 - macierze obrotu doprowadzajace kazda kamere do wspolnej plaszczyzny
      P1, P2 - macierze projekcji po rektyfikacji (zawieraja fx, fy, cx, cy)
      Q      - macierz reprojekcji 4x4: przelicza dysparycje na wspolrzedne 3D [mm]
    """
    left: CameraParams = field(default_factory=CameraParams)
    right: CameraParams = field(default_factory=CameraParams)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    T: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    E: np.ndarray = field(default_factory=lambda: np.eye(3))
    F: np.ndarray = field(default_factory=lambda: np.eye(3))
    reproj_error: float = 0.0
    # Wyniki stereoRectify - przechowywane, zeby rectify_maps() nie musial
    # wywolywac stereoRectify ponownie przy kazdym uzyciu
    R1: np.ndarray = field(default_factory=lambda: np.eye(3))
    R2: np.ndarray = field(default_factory=lambda: np.eye(3))
    P1: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))
    P2: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))
    Q: np.ndarray = field(default_factory=lambda: np.eye(4))  # disparity -> wspolrzedne 3D [mm]

    def rectify_maps(
        self, image_size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generuje mapy przeksztalcen do rektyfikacji pary stereo.



        Przyklad uzycia w petli przechwytywania:
            map1L, map2L, map1R, map2R = stereo.rectify_maps()
            left_rect  = cv2.remap(left_frame,  map1L, map2L, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_frame, map1R, map2R, cv2.INTER_LINEAR)

        Returns:
            (map1L, map2L, map1R, map2R) - mapy dla lewego i prawego obrazu
        """
        size = image_size or self.left.image_size
        # initUndistortRectifyMap laczy korekcje dystorsji i rektyfikacje w jednej mapie,
        # co pozwala wykonac obie operacje jednym wywolaniem cv2.remap()
        map1L, map2L = cv2.initUndistortRectifyMap(
            self.left.camera_matrix, self.left.dist_coeffs,
            self.R1, self.P1, size, cv2.CV_16SC2,
        )
        map1R, map2R = cv2.initUndistortRectifyMap(
            self.right.camera_matrix, self.right.dist_coeffs,
            self.R2, self.P2, size, cv2.CV_16SC2,
        )
        return map1L, map2L, map1R, map2R

    def to_dict(self):
        """Serializuje wszystkie parametry stereo do slownika (do zapisu JSON)."""
        return {
            "left": self.left.to_dict(), "right": self.right.to_dict(),
            "R": self.R.tolist(), "T": self.T.tolist(),
            "E": self.E.tolist(), "F": self.F.tolist(),
            "reproj_error": self.reproj_error,
            "R1": self.R1.tolist(), "R2": self.R2.tolist(),
            "P1": self.P1.tolist(), "P2": self.P2.tolist(),
            "Q":  self.Q.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        """Odtwarza obiekt z slownika wczytanego z pliku JSON.

        Klucze R1/R2/P1/P2/Q sa opcjonalne - starsze pliki kalibracji
        (zapisane przed dodaniem stereoRectify do pipeline) moga ich nie zawierac.
        W takim przypadku przyjmowane sa wartosci neutralne (macierze jednostkowe/zerowe)
        i rectify_maps() nie zadziala poprawnie - wymagana jest ponowna kalibracja.
        """
        return cls(
            left=CameraParams.from_dict(d["left"]),
            right=CameraParams.from_dict(d["right"]),
            R=np.array(d["R"]), T=np.array(d["T"]),
            E=np.array(d["E"]), F=np.array(d["F"]),
            reproj_error=d["reproj_error"],
            # Klucze R1/R2/P1/P2/Q sa opcjonalne - starsze pliki JSON moga ich nie miec
            R1=np.array(d.get("R1", np.eye(3))),
            R2=np.array(d.get("R2", np.eye(3))),
            P1=np.array(d.get("P1", np.zeros((3, 4)))),
            P2=np.array(d.get("P2", np.zeros((3, 4)))),
            Q=np.array(d.get("Q",  np.eye(4))),
        )


# ---------------------------------------------------------------------------
# Wykrywanie naroznikow szachownicy
# ---------------------------------------------------------------------------

def _board_points(is_landscape: bool = False) -> np.ndarray:
    """Generuje wspolrzedne 3D naroznikow szachownicy w ukladzie wzorca.

    Szachownica jest traktowana jako plaska (Z=0). Wspolrzedne X,Y sa
    wyrazone w milimetrach (numer_naroznika * SQUARE_SIZE_MM).
    Zwrocona tablica ma ksztalt (BOARD_ROWS*BOARD_COLS, 3).

    Przyklad dla szachownicy 3x3 z kwadratem 15mm:
        [(0,0,0), (15,0,0), (30,0,0),
         (0,15,0), (15,15,0), (30,15,0), ...]
    """
    pts = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    # mgrid generuje siatke indeksow, reshape(-1,2) spłaszcza do listy punktow
    if is_landscape:
        pts[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    else:
        pts[:, :2] = np.mgrid[0:BOARD_ROWS, 0:BOARD_COLS].T.reshape(-1, 2)
    return pts * SQUARE_SIZE


def find_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Wykrywa narozniki szachownicy na obrazie z precyzja subpikselowa.

    Detekcja odbywa sie na kopii zmniejszonej do CORNER_DETECT_MAX_WIDTH (szybkosc
    przy obrazach 3-4K z telefonow), ale wykryte narozniki sa NASTEPNIE przeskalowane
    z powrotem do wspolrzednych oryginalnego obrazu i dorefinowane subpikselowo na
    pelnej rozdzielczosci. Dzieki temu zwracane wspolrzedne sa zawsze w natywnej
    rozdzielczosci - kalibracja i image_size pozostaja spojne (zob. collect_points).

    Strategia dwuetapowa detekcji:
    1. findChessboardCornersSB - nowsza metoda z wbudowana precyzja subpikselowa,
       lepsza dla wysokich rozdzielczosci i trudnych warunkow oswietleniowych.
    2. findChessboardCorners + cornerSubPix - klasyczna metoda jako fallback,
       gdy SB nie znajdzie wzorca (np. czesciowe zasloniecie szachownicy).

    Args:
        image: obraz BGR lub grayscale

    Returns:
        tablica naroznikow (N,1,2) we wspolrzednych ORYGINALNEGO obrazu,
        lub None jezeli wzorzec nie zostal wykryty
    """
    # Konwersja do skali szarosci - algorytm detekcji naroznikow nie korzysta z koloru
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Detekcja na kopii zmniejszonej (jesli obraz jest szerszy niz prog).
    # scale = wspolczynnik z oryginalu do kopii roboczej; do przeliczenia narozników
    # z powrotem mnozymy przez 1/scale.
    h, w = gray_full.shape[:2]
    
    # Automatyczne dopasowanie siatki w zaleznosci od orientacji obrazu
    is_landscape = w > h
    if is_landscape:
        pattern_size = (BOARD_COLS, BOARD_ROWS)
    else:
        pattern_size = (BOARD_ROWS, BOARD_COLS)

    scale = 1.0
    gray = gray_full
    if w > config.CORNER_DETECT_MAX_WIDTH:
        scale = config.CORNER_DETECT_MAX_WIDTH / w
        gray = cv2.resize(gray_full, (config.CORNER_DETECT_MAX_WIDTH, int(round(h * scale))))

    # Metoda SB: dokladniejsza, wbudowana precyzja subpikselowa w jednym przejsciu
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not found:
        # Fallback: klasyczna metoda findChessboardCorners
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )
    if not found:
        return None

    # Przeskalowanie naroznikow z powrotem do wspolrzednych oryginalu i finalne
    # doprecyzowanie subpikselowe na pelnej rozdzielczosci (lepsza precyzja -> nizszy RMS)
    if scale != 1.0:
        corners = (corners.astype(np.float32) / scale)
    return cv2.cornerSubPix(
        gray_full, corners.astype(np.float32),
        config.CORNER_SUBPIX_WIN, config.CORNER_SUBPIX_ZERO_ZONE, CRITERIA,
    )


# ---------------------------------------------------------------------------
# Zbieranie punktow - kazdy obraz przetwarzany dokladnie raz
# ---------------------------------------------------------------------------

def _fix_corner_order(
    lc: np.ndarray, rc: np.ndarray, is_landscape: bool
) -> np.ndarray:
    """Corrects rc corner ordering so corners[i] matches the same physical board
    corner as lc[i]. Detects horizontal and/or vertical flip by comparing the
    board's row-direction and column-direction vectors between both cameras.
    Only the right-camera array is ever modified; lc is the reference.
    """
    cols = BOARD_COLS if is_landscape else BOARD_ROWS
    rows = BOARD_ROWS if is_landscape else BOARD_COLS

    # Direction along row 0 (first -> last corner in the row)
    row_dir_l = lc[cols - 1, 0] - lc[0, 0]
    row_dir_r = rc[cols - 1, 0] - rc[0, 0]
    h_flip = bool(np.dot(row_dir_l, row_dir_r) < 0)

    # Direction from row 0 to row 1 (column direction)
    col_dir_l = lc[cols, 0] - lc[0, 0]
    col_dir_r = rc[cols, 0] - rc[0, 0]
    v_flip = bool(np.dot(col_dir_l, col_dir_r) < 0)

    if not h_flip and not v_flip:
        return rc

    log.info(
        "Poprawiono kolejnosc naroznikow prawej kamery: "
        "h_flip=%s v_flip=%s - obrazy kamer maja rozna orientacje",
        h_flip, v_flip,
    )
    rc_grid = rc.reshape(rows, cols, 1, 2)
    if h_flip:
        rc_grid = rc_grid[:, ::-1, :, :]
    if v_flip:
        rc_grid = rc_grid[::-1, :, :, :]
    return rc_grid.reshape(-1, 1, 2)


def collect_points(image_paths: list[str]) -> CalibrationData:
    """Wczytuje obrazy i wykrywa narozniki szachownicy dla pojedynczej kamery.

    Kazdy obraz przetwarzany jest dokladnie raz. Wyniki nalezy przekazac
    bezposrednio do _calibrate_from_data(), aby uniknac ponownego wykrywania
    naroznikow (co jest operacja kosztowna obliczeniowo).

    Args:
        image_paths: lista sciezek do obrazow kalibracyjnych

    Returns:
        CalibrationData z zebranymi punktami 2D i 3D
    """
    obj_points, img_points, img_size = [], [], None
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            log.warning("Nie mozna wczytac: %s", path)
            continue
        # Rozmiar obrazu pobieramy z pierwszej poprawnie wczytanej klatki
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
            is_landscape = img_size[0] > img_size[1]
            objp = _board_points(is_landscape)
        corners = find_corners(img)
        if corners is None:
            log.warning("Brak naroznikow: %s", path)
            continue
        obj_points.append(objp)
        img_points.append(corners)
        log.info("OK: %s", path)
    log.info("Znaleziono wzorzec w %d/%d obrazach", len(img_points), len(image_paths))
    # Duzy odsetek odrzuconych klatek = problem z wzorcem (rozmiar BOARD_ROWS/COLS,
    # oswietlenie, rozmycie) - sygnalizujemy, bo czesto to przyczyna slabej kalibracji.
    if image_paths and len(img_points) < 0.5 * len(image_paths):
        log.warning("Odrzucono %d/%d klatek - sprawdz rozmiar szachownicy (%dx%d), "
                    "oswietlenie i ostrosc", len(image_paths) - len(img_points),
                    len(image_paths), BOARD_ROWS, BOARD_COLS)
    if not img_points:
        raise ValueError("Nie wykryto wzorca na zadnym obrazie")
    assert img_size is not None  # ustawiane przy pierwszym poprawnym obrazie
    return CalibrationData(obj_points, img_points, img_size)


def collect_stereo_points(
    left_paths: list[str], right_paths: list[str],
    debug_dir: "Path | None" = None,
) -> StereoCalibrationData:
    """Wykrywa narozniki szachownicy w parach stereo.

    Kazdy obraz przetwarzany jest dokladnie raz. Do wyniku trafiaja tylko
    pary, gdzie OBIE kamery wykryly wzorzec - stereoCalibrate wymaga
    pelnych, odpowiadajacych sobie punktow z obu kamer.

    Args:
        left_paths:  posortowane sciezki do obrazow lewej kamery
        right_paths: posortowane sciezki do obrazow prawej kamery (ta sama kolejnosc)

    Returns:
        StereoCalibrationData z dopasowanymi parami punktow
    """
    obj_pts, left_pts, right_pts, img_size = [], [], [], None
    for lp, rp in zip(left_paths, right_paths):
        l_img, r_img = cv2.imread(lp), cv2.imread(rp)
        if l_img is None or r_img is None:
            log.warning("Nie mozna wczytac pary: %s, %s", lp, rp)
            continue

        # Kalibracja w natywnej rozdzielczosci: find_corners samo zmniejsza obraz
        # tylko na czas detekcji i zwraca narozniki w pelnej skali. image_size
        # bierzemy z lewego obrazu (stereoCalibrate uzywa jednego imageSize) -
        # zakladamy, ze obie kamery maja te sama rozdzielczosc.
        if img_size is None:
            img_size = (l_img.shape[1], l_img.shape[0])
            is_landscape = img_size[0] > img_size[1]
            objp = _board_points(is_landscape)
        if (r_img.shape[1], r_img.shape[0]) != (l_img.shape[1], l_img.shape[0]):
            log.warning("Rozne rozdzielczosci w parze (%s vs %s): %s, %s",
                        (l_img.shape[1], l_img.shape[0]),
                        (r_img.shape[1], r_img.shape[0]), lp, rp)

        lc, rc = find_corners(l_img), find_corners(r_img)
        if lc is None or rc is None:
            # Para jest niekompletna - odrzucamy calosc, nie mozna uzyc czesciowych danych
            log.warning("Brak naroznikow w parze: %s, %s", lp, rp)
            continue
        is_landscape_img = l_img.shape[1] > l_img.shape[0]
        rc = _fix_corner_order(lc, rc, is_landscape_img)
        obj_pts.append(objp)
        left_pts.append(lc)
        right_pts.append(rc)
        log.info("OK para: %s | %s", lp, rp)

        if debug_dir is not None:
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                pair_idx = len(obj_pts) - 1
                l_dbg = l_img.copy()
                r_dbg = r_img.copy()
                p_size = (BOARD_COLS, BOARD_ROWS) if is_landscape_img else (BOARD_ROWS, BOARD_COLS)
                cv2.drawChessboardCorners(l_dbg, p_size, lc, True)
                cv2.drawChessboardCorners(r_dbg, p_size, rc, True)
                combined = np.hstack([l_dbg, r_dbg])
                cv2.imwrite(str(debug_dir / f"pair_{pair_idx:04d}.png"), combined)
            except Exception as _e:
                log.debug("Debug corner image save failed: %s", _e)
    log.info("Zgodnych par: %d/%d", len(obj_pts), len(left_paths))
    # Para jest uzyteczna tylko gdy OBIE kamery widza wzorzec - duzo niekompletnych
    # par oznacza zly kadr jednej z kamer lub niezsynchronizowane ujecia.
    if left_paths and len(obj_pts) < 0.5 * len(left_paths):
        log.warning("Odrzucono %d/%d par (wzorzec niewidoczny w obu kamerach) - "
                    "sprawdz czy szachownica miesci sie w obu kadrach",
                    len(left_paths) - len(obj_pts), len(left_paths))
    if not obj_pts:
        raise ValueError("Nie wykryto wzorca na zadnej parze obrazow")
    assert img_size is not None
    return StereoCalibrationData(obj_pts, left_pts, right_pts, img_size)


# ---------------------------------------------------------------------------
# Kalibracja
# ---------------------------------------------------------------------------

def _calibrate_from_data(data: CalibrationData) -> CameraParams:
    """Kalibruje kamere na podstawie juz zebranych punktow (bez ponownego I/O).

    Wywoluje cv2.calibrateCamera, ktore metoda Zhanga wyznacza macierz wewnetrzna
    K i wspolczynniki dystorsji minimalizujac blad reprojekcji (RMS [px]).
    Jako punkt startowy przekazujemy macierz jednostkowa i zerowe dystorsje -
    OpenCV sam wyznacza poczatkowe przyblizenie metoda DLT.


    """
    if len(data) < config.MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Za malo obrazow z wykrytym wzorcem ({len(data)}), "
            f"min. {config.MIN_CALIBRATION_IMAGES}"
        )
    # CALIB_FIX_K3 prevents k1/k2/k3 from fighting each other (overfitting).
    # Phone cameras have near-zero higher-order distortion; letting k3 float
    # causes k2 and k3 to take extreme opposite values that cancel on training
    # data but destroy undistortion maps (valid area shrinks to a tiny oval).
    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        data.obj_points, data.img_points, data.image_size, np.eye(3), np.zeros(5),
        flags=cv2.CALIB_FIX_K3,
    )
    log.info("RMS reproj. error: %.4f px (%d klatek, rozdz. %dx%d)",
             rms, len(data), data.image_size[0], data.image_size[1])
    # Wysoki RMS pojedynczej kamery (>1 px) zwykle oznacza nieostre zdjecia,
    # zle wykryte narozniki lub zbyt malo roznorodnych poz szachownicy.
    if rms > 1.0:
        log.warning("Wysoki RMS kalibracji kamery: %.4f px (>1 px) - sprawdz ostrosc "
                    "i roznorodnosc ujec szachownicy", rms)
    return CameraParams(
        camera_matrix=mtx, dist_coeffs=dist, reproj_error=rms, image_size=data.image_size
    )


def calibrate_single(image_paths: list[str]) -> CameraParams:
    """Kalibruje pojedyncza kamere ze sciezek do obrazow.
    Publiczny wrapper laczacy collect_points() i _calibrate_from_data()."""
    return _calibrate_from_data(collect_points(image_paths))


def baseline_warning(T: np.ndarray, rms: float) -> Optional[str]:
    """Zwraca ostrzezenie diagnostyczne o jakosci geometrii stereo lub None.

    Wykrywa dwa typowe problemy sesji pomiarowej:
      1. Baza nie-pozioma: poprawne ustawienie to T ~ [|T|, 0, 0] (telefony w jednej
         poziomej linii). Duze skladowe Y/Z oznaczaja, ze jeden telefon byl wysuniety
         do przodu lub w gore - linie epipolarne nie sa poziome, co psuje SGBM.
      2. Wysoki RMS stereo mimo dobrych kalibracji indywidualnych - zwykle oznacza,
         ze telefony poruszyly sie miedzy klatkami (brak sztywnosci ukladu).
    """
    t = np.asarray(T, dtype=float).flatten()
    tx, ty, tz = abs(t[0]), abs(t[1]), abs(t[2])
    msgs = []
    if tx > 1e-6 and max(ty, tz) > 0.2 * tx:
        msgs.append(
            f"baza stereo nie jest pozioma: T=[{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}] mm "
            f"(oczekiwane ~[{np.linalg.norm(t):.0f}, 0, 0]) - sprawdz ustawienie telefonow"
        )
    if rms > config.MAX_STEREO_REPROJ_ERROR:
        msgs.append(
            f"stereo RMS={rms:.2f} px > prog {config.MAX_STEREO_REPROJ_ERROR} px - "
            f"telefony mogly poruszyc sie miedzy klatkami lub potrzeba wiecej par"
        )
    return "; ".join(msgs) if msgs else None


def calibrate_stereo(
    left_paths: list[str], right_paths: list[str],
    debug_dir: "Path | None" = None,
) -> StereoParams:
    """Kalibruje pare stereo i wyznacza wszystkie parametry potrzebne do pomiaru.

    Etapy:
      1. collect_stereo_points  - wykrywa narozniki w parach (kazdy obraz raz)
      2. _calibrate_from_data   - kalibruje lewa i prawa kamere osobno
      3. cv2.stereoCalibrate    - wyznacza wzajemne polozenie kamer (R, T, E, F)
      4. cv2.stereoRectify      - oblicza macierze R1/R2/P1/P2/Q do rektyfikacji

    Flaga CALIB_FIX_INTRINSIC w stereoCalibrate oznacza, ze parametry wewnetrzne
    kamer (z kroku 2) sa traktowane jako stale - tylko R i T sa optymalizowane.
    Dzieki temu stereoCalibrate jest stabilniejszy numerycznie.

    Args:
        left_paths:  posortowane sciezki do obrazow lewej kamery
        right_paths: posortowane sciezki do obrazow prawej kamery

    Returns:
        StereoParams gotowe do uzycia w rectify_maps() i dalszym pipeline
    """
    if len(left_paths) != len(right_paths):
        raise ValueError("Liczba obrazow lewej i prawej kamery musi byc rowna")

    stereo_data = collect_stereo_points(left_paths, right_paths, debug_dir=debug_dir)
    if len(stereo_data) < config.MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Za malo par z wzorcem ({len(stereo_data)}), "
            f"min. {config.MIN_CALIBRATION_IMAGES}"
        )

    log.info("Kalibracja stereo na %d parach...", len(stereo_data))
    # Uzywamy tych samych wykrytych punktow do kalibracji indywidualnych kamer
    # i do stereoCalibrate - eliminuje ponowne I/O i gwarantuje spojnosc danych
    left_cam = _calibrate_from_data(stereo_data.left)
    right_cam = _calibrate_from_data(stereo_data.right)

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        stereo_data.obj_points,
        stereo_data.left_points,
        stereo_data.right_points,
        left_cam.camera_matrix, left_cam.dist_coeffs,
        right_cam.camera_matrix, right_cam.dist_coeffs,
        stereo_data.image_size, criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    log.info(
        "RMS reproj.: lewa=%.4f px, prawa=%.4f px, stereo=%.4f px (rozdz. %dx%d)",
        left_cam.reproj_error, right_cam.reproj_error, rms,
        stereo_data.image_size[0], stereo_data.image_size[1],
    )
    warn = baseline_warning(T, rms)
    if warn:
        log.warning("Jakosc kalibracji stereo: %s", warn)

    # stereoRectify wyznacza macierze R1/R2/P1/P2/Q potrzebne do rektyfikacji.
    # alpha=0: przycina wynik do czesci wspolnej obu kamer (zero czarnych pikseli),
    # ogniskowa bliska oryginalnej. Dziala poprawnie gdy dystorsja jest sensowna
    # (CALIB_FIX_K3 pilnuje, ze tak jest). alpha=1 przy pochylonych kamerach
    # potrafi skompresowac ogniskowa 24x (np. 1400->58 px), bo musi objac
    # pelne FOV obu kamer po duzej rotacji rektyfikacyjnej.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_cam.camera_matrix, left_cam.dist_coeffs,
        right_cam.camera_matrix, right_cam.dist_coeffs,
        stereo_data.image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    # Sanity check: rectified focal length should be comparable to input camera focal lengths.
    # If it's >3x the average, stereoRectify received bad input (high RMS, few/poor images).
    f_avg = (left_cam.camera_matrix[0, 0] + right_cam.camera_matrix[0, 0]) / 2.0
    f_rect = float(P1[0, 0])
    if f_rect > f_avg * 3.0:
        log.warning(
            "stereoRectify zwrocil ogniskowa %.0f px (%.1fx wieksza niz srednia kamer %.0f px) — "
            "kalibracja stereo jest bledna. Przyczyny: zbyt malo par kalibracyjnych (%d), "
            "zle oswietlenie/ostrosc, lub telefony za bardzo pochylone. "
            "Dodaj wiecej par (>=10) z roznych katow i odleglosci szachownicy.",
            f_rect, f_rect / f_avg, f_avg, len(stereo_data),
        )

    return StereoParams(
        left=left_cam, right=right_cam,
        R=R, T=T, E=E, F=F, reproj_error=rms,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    )


# ---------------------------------------------------------------------------
# Zapis / odczyt parametrow
# ---------------------------------------------------------------------------

def save_params(params: CameraParams | StereoParams, path: str):
    """Zapisuje parametry kalibracji do pliku JSON.
    Tworzy katalogi nadrzedne jesli nie istnieja."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params.to_dict(), f, indent=2)
    log.info("Zapisano: %s", path)


@overload
def load_params(path: str, stereo: Literal[False] = False) -> CameraParams: ...


@overload
def load_params(path: str, stereo: Literal[True]) -> StereoParams: ...


def load_params(path: str, stereo: bool = False) -> CameraParams | StereoParams:
    """Wczytuje parametry kalibracji z pliku JSON.

    Args:
        path:   sciezka do pliku JSON (zapisanego przez save_params)
        stereo: True -> zwraca StereoParams, False -> zwraca CameraParams
    """
    with open(path) as f:
        d = json.load(f)
    return StereoParams.from_dict(d) if stereo else CameraParams.from_dict(d)


def get_image_paths(directory: str, pattern: str | None = None) -> list[str]:
    """Zwraca posortowana liste sciezek do obrazow w podanym katalogu.

    Przeszukuje kolejno rozszerzenia z config.IMAGE_EXTENSIONS i zwraca
    pierwsze niepuste dopasowanie. Jezeli pattern jest podany, uzywa tylko jego.
    """
    exts = [pattern] if pattern else config.IMAGE_EXTENSIONS
    for ext in exts:
        paths = sorted(glob.glob(str(Path(directory) / ext)))
        if paths:
            return paths
    return []


# ---------------------------------------------------------------------------
# CLI - uruchomienie jako skrypt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kalibracja kamery/stereo")
    parser.add_argument("--mode", choices=["single", "stereo"], default="single")
    parser.add_argument("--left-dir", default=os.path.join(CALIB_DIR, "left"))
    parser.add_argument("--right-dir", default=os.path.join(CALIB_DIR, "right"))
    parser.add_argument("--output", default=os.path.join(CALIB_OUT, "calibration.json"))
    args = parser.parse_args()

    if args.mode == "single":
        # Dla trybu single szukamy obrazow najpierw w left-dir, potem w CALIB_DIR
        imgs = get_image_paths(args.left_dir) or get_image_paths(CALIB_DIR)
        params = calibrate_single(imgs)
    else:
        params = calibrate_stereo(
            get_image_paths(args.left_dir),
            get_image_paths(args.right_dir),
        )
    save_params(params, args.output)