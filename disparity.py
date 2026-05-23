# disparity.py
"""Mapa dysparycji i mapa glebokosci z pary stereo.

Przepływ przetwarzania:
  1. rectify_pair()        - korekcja dystorsji + wyrownanie linii epipolarnych
  2. compute_disparity()   - SGBM: obliczenie przesuniecia pikseli miedzy obrazami
  3. disparity_to_depth()  - przeliczenie dysparycji na glebokos [mm] przez macierz Q

Uzycie CLI:
    python disparity.py --calib calib_output/stereo.json --left left.png --right right.png

Z kodu:
    from calibration import load_params
    from disparity import rectify_pair, compute_disparity, disparity_to_depth

    stereo = load_params("calib_output/stereo.json", stereo=True)
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img)
    disp = compute_disparity(left_rect, right_rect)
    depth = disparity_to_depth(disp, stereo.Q)
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from calibration import StereoParams, load_params

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domyslne parametry SGBM - dzialaja dobrze dla telefonow ~10-20 cm bazy
# ---------------------------------------------------------------------------
@dataclass
class SGBMConfig:
    """Parametry Semi-Global Block Matching.

    min_disparity: zwykle 0, chyba ze obiekty wychodza poza lewy kadr
    num_disparities: wielokrotnosc 16; 64 dla bliskich objetkow, 128+ dla dalszych
    block_size:      nieparzysta, 5-11; mniejsza = wiecej szumu, wieksza = rozmycie krawedzi
    p1, p2:          kary za nieciaglosci; p2 >= 4*p1 to dobry punkt startowy
    disp12_max_diff: sprawdzenie L-R konsystencji; 1-2 eliminuje bledy na krawedziach
    uniqueness:      5-15; wyzsze = ostrозniejsze dopasowanie, wiecej dziur
    speckle_window:  50-200; filtr malych "plamek" bledu
    speckle_range:   1-2
    """
    min_disparity: int = 0
    num_disparities: int = 64
    block_size: int = 7
    p1: int = 8 * 3 * 7 ** 2      # 8 * kanaly * block_size^2
    p2: int = 32 * 3 * 7 ** 2     # 32 * kanaly * block_size^2
    disp12_max_diff: int = 1
    uniqueness: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


# ---------------------------------------------------------------------------
# Rektyfikacja stereo
# ---------------------------------------------------------------------------

def rectify_pair(
    stereo: StereoParams,
    left: np.ndarray,
    right: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rektyfikuje pare stereo uzywajac parametrow z calibration.py.

    Wywoluje stereo.rectify_maps() i aplikuje cv2.remap na obu obrazach.
    Po rektyfikacji odpowiadajace sobie punkty leza na tych samych
    poziomych liniach epipolarnych - warunek konieczny dla poprawnego
    dzialania algorytmu SGBM.

    Mapy rektyfikacji (a wiec i macierz Q) sa wyznaczone dla rozdzielczosci uzytej
    podczas kalibracji (stereo.left.image_size). Jezeli obrazy wejsciowe maja inna
    rozdzielczosc, sa automatycznie skalowane do rozmiaru kalibracji - dzieki temu
    macierz Q i mapy zawsze pasuja (bez tego glebia jest bledna, co wymagalo
    wczesniej recznej korekty Q).

    Args:
        stereo:     StereoParams z load_params(..., stereo=True)
        left:       obraz lewej kamery (BGR lub grayscale)
        right:      obraz prawej kamery
        image_size: (width, height) docelowy rozmiar rektyfikacji; jesli None,
                    brany z stereo.left.image_size (rozdzielczosc kalibracji)

    Returns:
        (left_rect, right_rect) - rektyfikowane obrazy
    """
    size = image_size or stereo.left.image_size

    # Obrazy wejsciowe musza miec rozdzielczosc zgodna z mapami rektyfikacji.
    # Jezeli sie roznia (np. zdjecie pomiarowe w innej rozdzielczosci niz kalibracja),
    # skalujemy je do rozmiaru kalibracji, zeby macierz Q pozostala poprawna.
    def _match(img: np.ndarray, name: str) -> np.ndarray:
        if (img.shape[1], img.shape[0]) != tuple(size):
            log.warning("Rozdzielczosc %s %s != rozmiar kalibracji %s - skaluje",
                        name, (img.shape[1], img.shape[0]), tuple(size))
            return cv2.resize(img, tuple(size))
        return img

    left  = _match(left,  "lewego obrazu")
    right = _match(right, "prawego obrazu")

    map1L, map2L, map1R, map2R = stereo.rectify_maps(size)
    left_rect  = cv2.remap(left,  map1L, map2L, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map1R, map2R, cv2.INTER_LINEAR)
    log.info("Rektyfikacja OK: %s -> %s", left.shape[:2], left_rect.shape[:2])
    return left_rect, right_rect


# ---------------------------------------------------------------------------
# Mapa dysparycji - SGBM
# ---------------------------------------------------------------------------

def compute_disparity(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    cfg: SGBMConfig | None = None,
) -> np.ndarray:
    """Oblicza mape dysparycji metoda SGBM.

    Dysparycja to przesuniecie piksela miedzy lewym a prawym obrazem [px].
    Wieksza dysparycja = obiekt blizej kamery.

    Args:
        left_rect:  rektyfikowany obraz lewy (BGR lub grayscale)
        right_rect: rektyfikowany obraz prawy
        cfg:        parametry SGBM; jesli None, uzywa SGBMConfig()

    Returns:
        disp_float: mapa dysparycji float32 w pikselach (nieprawidlowe=0)
    """
    if cfg is None:
        cfg = SGBMConfig()

    # SGBM operuje na jednokanałowych obrazach szaroskalowych - kolor nie wnosi
    # dodatkowej informacji do porownywania bloków pikseli miedzy obrazami
    gray_l = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY) if left_rect.ndim  == 3 else left_rect
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY) if right_rect.ndim == 3 else right_rect

    matcher = cv2.StereoSGBM_create(
        minDisparity=cfg.min_disparity,
        numDisparities=cfg.num_disparities,
        blockSize=cfg.block_size,
        P1=cfg.p1,
        P2=cfg.p2,
        disp12MaxDiff=cfg.disp12_max_diff,
        uniquenessRatio=cfg.uniqueness,
        speckleWindowSize=cfg.speckle_window,
        speckleRange=cfg.speckle_range,
        mode=cfg.mode,
    )

    # OpenCV zwraca dysparycje * 16 (zapis stałoprzecinkowy dla precyzji),
    # dzielenie przez 16 przywraca wartosc w pikselach z dokladnoscia 1/16 px
    disp_raw = matcher.compute(gray_l, gray_r)
    disp_float = disp_raw.astype(np.float32) / 16.0

    # Piksele z dysparycja <= 0 oznaczaja brak dopasowania po stronie lewej
    # lub prawej - zerujemy je, aby nie powodowaly bledow przy konwersji do glebokosci
    invalid = (disp_float <= 0) | (disp_float >= cfg.num_disparities)
    disp_float[invalid] = 0.0

    valid_px = int((disp_float > 0).sum())
    total_px = disp_float.size
    log.info("Dysparycja: zakres [%.1f, %.1f] px, waznych pikseli: %d/%d (%.0f%%)",
             float(disp_float[disp_float > 0].min()) if valid_px else 0,
             float(disp_float.max()),
             valid_px, total_px, 100 * valid_px / total_px)
    return disp_float


# ---------------------------------------------------------------------------
# Mapa glebokosci - reprojekcja dysparycji w przestrzen 3D
# ---------------------------------------------------------------------------

def disparity_to_depth(
    disparity: np.ndarray,
    Q: np.ndarray,
    max_depth_mm: float = 5000.0,
) -> np.ndarray:
    """Przelicza dysparycje na glebokos w milimetrach uzywajac macierzy Q.

    Q to macierz 4x4 z cv2.stereoRectify, zapisana w StereoParams.Q.
    cv2.reprojectImageTo3D mnozy [x, y, d, 1]^T przez Q i daje [X, Y, Z, W];
    glebokos = Z/W w jednostkach bazy (mm jesli SQUARE_SIZE_MM w mm).

    Dzieki tej operacji kazdy piksel mapy dysparycji jest bezposrednio
    zamieniany na wspolrzedna glebokosci w przestrzeni 3D.

    Args:
        disparity:    mapa dysparycji float32 (z compute_disparity)
        Q:            macierz stereo.Q z StereoParams
        max_depth_mm: piksele glébsze niz ten prog sa zerowane (odfiltrowanie szumu)

    Returns:
        depth_mm: mapa glebokosci float32 [mm], 0 = brak danych
    """
    points_3d = cv2.reprojectImageTo3D(disparity, Q)   # (H, W, 3) -> X, Y, Z [mm]
    depth_mm = points_3d[:, :, 2].copy()

    # Zerujemy piksele bez dysparycji oraz te poza zakresem pomiarowym.
    # Wartosc 0 sluzy jako znacznik "brak danych" na mapie glebokosci.
    no_data = (disparity <= 0) | (depth_mm <= 0) | (depth_mm > max_depth_mm)
    depth_mm[no_data] = 0.0

    valid = depth_mm[depth_mm > 0]
    if valid.size > 0:
        log.info("Glebokos: min=%.0f mm, max=%.0f mm, mediana=%.0f mm",
                 float(valid.min()), float(valid.max()), float(np.median(valid)))
    return depth_mm.astype(np.float32)


# ---------------------------------------------------------------------------
# Wizualizacje
# ---------------------------------------------------------------------------

def colormap_disparity(disparity: np.ndarray) -> np.ndarray:
    """Zwraca kolorowa mape dysparycji (BGR) do zapisu lub wyswietlenia.

    Normalizuje wartosci tylko wsrod waznych pikseli (dysparycja > 0),
    zeby nie zafalsowac skali przez piksele bez danych.
    Czarne piksele = brak dopasowania SGBM.
    """
    d = disparity.copy()
    mask = d > 0
    if mask.any():
        # Normalizacja w zakresie waznych pikseli - ignorujemy zera (brak danych)
        d[mask] = cv2.normalize(d[mask].reshape(-1, 1), None, 0, 255,
                                cv2.NORM_MINMAX).flatten()
    d8 = d.astype(np.uint8)
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    colored[~mask] = 0   # czarne piksele = brak danych
    return colored


def colormap_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Zwraca kolorowa mape glebokosci (BGR) do zapisu lub wyswietlenia.

    Skala jest odwrocona (255 - wartość) zeby blizsze obiekty byly cieplejsze
    (czerwone), a dalsze zimniejsze (niebieskie).
    Czarne piksele = brak danych.
    """
    d = depth_mm.copy()
    mask = d > 0
    if mask.any():
        d[mask] = cv2.normalize(d[mask].reshape(-1, 1), None, 0, 255,
                                cv2.NORM_MINMAX).flatten()
    # Odwrocenie skali: blizej = cieplej (wyzsza wartosc po odwroceniu)
    d8 = (255 - d.astype(np.uint8))
    d8[~mask] = 0
    colored = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    colored[~mask] = 0
    return colored


def draw_epipolar_check(left_rect: np.ndarray, right_rect: np.ndarray,
                        n_lines: int = 12) -> np.ndarray:
    """Laczy rektyfikowane obrazy i rysuje linie epipolarne do weryfikacji.

    Jesli rektyfikacja jest poprawna, charakterystyczne punkty na obu obrazach
    powinny lezec na tej samej poziomej linii.
    """
    combined = np.hstack([left_rect, right_rect])
    h = combined.shape[0]
    step = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * step
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 200, 255), 1, cv2.LINE_AA)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Mapa dysparycji i glebokosci")
    parser.add_argument("--calib",  default="calib_output/stereo.json",
                        help="JSON z kalibracja stereo (z calibration.py)")
    parser.add_argument("--left",   required=True, help="Obraz lewej kamery")
    parser.add_argument("--right",  required=True, help="Obraz prawej kamery")
    parser.add_argument("--output", default="./depth_output",
                        help="Katalog na wyniki")
    parser.add_argument("--max-depth", type=float, default=5000.0,
                        help="Maks. glebokos [mm] do wizualizacji")
    args = parser.parse_args()

    stereo = load_params(args.calib, stereo=True)

    left_img  = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    if left_img is None or right_img is None:
        raise SystemExit("Nie mozna wczytac obrazow")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Kolejnosc: rektyfikacja -> dysparycja -> glebokos -> zapis wynikow
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img)
    disp   = compute_disparity(left_rect, right_rect)
    depth  = disparity_to_depth(disp, stereo.Q, args.max_depth)

    # Zapis wszystkich wynikow - obrazy kolorowe do podgladu, .npy do dalszego przetwarzania
    cv2.imwrite(str(out / "epipolar_check.png"),  draw_epipolar_check(left_rect, right_rect))
    cv2.imwrite(str(out / "disparity_color.png"), colormap_disparity(disp))
    cv2.imwrite(str(out / "depth_color.png"),     colormap_depth(depth))
    np.save(str(out / "disparity.npy"), disp)
    np.save(str(out / "depth_mm.npy"),  depth)

    print(f"Wyniki zapisane do: {out}/")