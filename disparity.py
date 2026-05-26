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

import config
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
    num_disparities: int = config.SGBM_NUM_DISPARITIES
    block_size: int = 7
    p1: int = 8 * 3 * 7 ** 2      # 8 * kanaly * block_size^2
    p2: int = 32 * 3 * 7 ** 2     # 32 * kanaly * block_size^2
    disp12_max_diff: int = 1
    uniqueness: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM


def auto_sgbm_cfg(
    stereo: "StereoParams",
    img_w: int,
    max_depth_mm: float = 5000.0,
    min_depth_mm: float = 800.0,
) -> "SGBMConfig":
    """Compute SGBM min_disparity + num_disparities from calibration geometry.

    SGBM only produces valid output for columns x >= min_disp + num_disp.
    This function sets:
      min_disp = disparity at max_depth_mm  (ignore far background)
      num_disp = covers down to min_depth_mm (ensure near objects are matched),
                 capped at img_w so we don't exceed image width.

    Formula: d = f * B / Z
      f = P1[0,0]            (rectified focal length, px)
      B = |P2[0,3]| / f      (horizontal baseline, mm)
    """
    f = float(abs(stereo.P1[0, 0]))
    tx = float(abs(stereo.P2[0, 3]))
    if f < 1.0 or tx < 1.0:
        log.warning("auto_sgbm_cfg: P1/P2 nie wygladaja na poprawne - uzywam domyslnego SGBMConfig")
        return SGBMConfig()

    # Sanity-check rectified focal length against individual camera focal lengths.
    # stereoRectify can return absurd values (10-100x too large) when the stereo
    # calibration is poor (high RMS, bad checkerboard coverage). When detected,
    # fall back to the average of individual camera focal lengths and estimate
    # the baseline directly from |T| — this gives reasonable SGBM parameters
    # even with a bad calibration, though measurement accuracy will still be limited.
    f_left = float(abs(stereo.left.camera_matrix[0, 0]))
    f_right = float(abs(stereo.right.camera_matrix[0, 0]))
    f_raw_avg = (f_left + f_right) / 2.0
    if f > f_raw_avg * 3.0:
        log.warning(
            "auto_sgbm_cfg: P1[0,0]=%.0f px jest %.1fx wieksza niz ogniskowe kamer "
            "(lewa=%.0f, prawa=%.0f px) — stereoRectify zwrocil bledne parametry "
            "(stereo RMS=%.2f px). Uzywam sredniej ogniskowej kamer (%.0f px) "
            "i normy |T| jako bazy; pomiar bedzie niedokladny do czasu rekalibracji.",
            f, f / f_raw_avg, f_left, f_right, stereo.reproj_error, f_raw_avg,
        )
        f = f_raw_avg
        B_phys = float(np.linalg.norm(stereo.T))  # physical baseline magnitude [mm]
        tx = f * B_phys

    B = tx / f  # baseline in mm

    # Minimum detectable distance = closest object whose full disparity fits in img_w
    min_detectable_mm = f * B / img_w
    if min_detectable_mm > 1500.0:
        log.warning(
            "auto_sgbm_cfg: minimalna wykrywalna odleglosc = %.0f mm "
            "(f=%.0f px, B=%.0f mm, szerokosc=%d px) - "
            "baza stereo jest za duza dla tej odleglosci; "
            "zmniejsz baze do ~%.0f mm lub odsun kamery od obiektu",
            min_detectable_mm, f, B, img_w,
            img_w * 1000.0 / f,
        )

    # min_disparity: background objects at max_depth (floor to 16-multiple)
    d_far = max(0, int(f * B / max_depth_mm))
    min_disp = (d_far // 16) * 16

    # num_disparities: cover from max_depth down to min_depth.
    # d_near may exceed img_w for large baselines — cap at img_w so we at least
    # match whatever is physically visible in both images.
    d_near = int(f * B / min_depth_mm)
    d_near_capped = min(d_near, img_w)
    num_disp = max(16, ((d_near_capped - min_disp + 15) // 16) * 16)

    valid_start = min_disp + num_disp
    coverage_pct = 100.0 * max(0, img_w - valid_start) / img_w

    if coverage_pct < 20.0:
        log.warning(
            "auto_sgbm_cfg: pokrycie dysparycji tylko %.0f%% przy odleglosci %.0f mm "
            "(valid x>=%d z %d px szerokosci) - zmniejsz baze z %.0f mm do ~%.0f mm "
            "aby uzyskac >50%% pokrycia",
            coverage_pct, min_depth_mm, valid_start, img_w,
            B, img_w * min_depth_mm / (2 * f),
        )

    cfg = SGBMConfig()
    cfg.min_disparity = min_disp
    cfg.num_disparities = num_disp
    log.info(
        "auto_sgbm_cfg: f=%.0fpx B=%.0fmm -> min_disp=%d num_disp=%d "
        "(valid x>=%d, pokrycie ~%.0f%% przy min_depth=%.0fmm)",
        f, B, min_disp, num_disp,
        valid_start, coverage_pct, min_depth_mm,
    )
    return cfg


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

    img_w = left_rect.shape[1]

    # OpenCV wymaga: width > minDisparity + numDisparities + blockSize/2.
    # Jesli num_disparities jest za duze dla szerokosci obrazu, zmniejszamy je
    # do najblizszej wielokrotnosci 16 spelniajace ten warunek.
    max_safe_num_disp = ((img_w - cfg.min_disparity - cfg.block_size // 2 - 1) // 16) * 16
    num_disp = min(cfg.num_disparities, max(16, max_safe_num_disp))
    if num_disp != cfg.num_disparities:
        log.warning("SGBM: num_disparities %d zbyt duze dla szerokosci obrazu %d px "
                    "- zmniejszono do %d (max bezpieczne)",
                    cfg.num_disparities, img_w, num_disp)

    log.debug("SGBM cfg: num_disp=%d block=%d uniq=%d speckle_win=%d mode=%d",
              num_disp, cfg.block_size, cfg.uniqueness, cfg.speckle_window, cfg.mode)

    # SGBM wymaga identycznych rozmiarow obrazow - rozne rozmiary daja smieciowa
    # dysparycje lub blad OpenCV. Ostrzegamy, bo to typowy efekt zlej rektyfikacji.
    if left_rect.shape[:2] != right_rect.shape[:2]:
        log.warning("SGBM: rozne rozmiary obrazow L=%s R=%s - dysparycja bedzie bledna",
                    left_rect.shape[:2], right_rect.shape[:2])

    # SGBM operuje na jednokanałowych obrazach szaroskalowych - kolor nie wnosi
    # dodatkowej informacji do porownywania bloków pikseli miedzy obrazami
    gray_l = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY) if left_rect.ndim  == 3 else left_rect
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY) if right_rect.ndim == 3 else right_rect

    matcher = cv2.StereoSGBM_create( # type: ignore
        minDisparity=cfg.min_disparity,
        numDisparities=num_disp,
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
    invalid = (disp_float <= 0) | (disp_float >= num_disp)
    disp_float[invalid] = 0.0

    valid_px = int((disp_float > 0).sum())
    total_px = disp_float.size
    coverage = 100 * valid_px / total_px
    log.info("Dysparycja: zakres [%.1f, %.1f] px, waznych pikseli: %d/%d (%.0f%%)",
             float(disp_float[disp_float > 0].min()) if valid_px else 0,
             float(disp_float.max()),
             valid_px, total_px, coverage)
    if valid_px == 0:
        log.warning("Dysparycja: 0 waznych pikseli - brak dopasowan SGBM "
                    "(sprawdz rektyfikacje, oswietlenie, num_disparities=%d)", cfg.num_disparities)
    elif coverage < 10.0:
        log.warning("Dysparycja: niskie pokrycie %.0f%% - gladka/jednolita scena lub "
                    "zla rektyfikacja; chmura punktow bedzie rzadka", coverage)
    # Wartosci blisko gornego progu sugeruja, ze obiekty sa blizej niz zaklada
    # num_disparities - czesc dysparycji moze byc obcieta (utrata bliskich punktow).
    if valid_px and float(disp_float.max()) >= num_disp - 1:
        log.warning("Dysparycja: max=%.1f px blisko limitu num_disparities=%d - "
                    "bliskie obiekty moga byc obciete, rozwaz wieksze num_disparities",
                    float(disp_float.max()), num_disp)
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

    # Liczymy ile pikseli odpada na kazdym kroku - pomaga zdiagnozowac,
    # czy dane gina przez brak dysparycji, czy przez prog max_depth_mm.
    n_disp = int((disparity > 0).sum())
    too_far = int(((disparity > 0) & (depth_mm > max_depth_mm)).sum())

    # Zerujemy piksele bez dysparycji oraz te poza zakresem pomiarowym.
    # Wartosc 0 sluzy jako znacznik "brak danych" na mapie glebokosci.
    no_data = (disparity <= 0) | (depth_mm <= 0) | (depth_mm > max_depth_mm)
    depth_mm[no_data] = 0.0

    valid = depth_mm[depth_mm > 0]
    if valid.size > 0:
        log.info("Glebokos: min=%.0f mm, max=%.0f mm, mediana=%.0f mm (%d pkt)",
                 float(valid.min()), float(valid.max()), float(np.median(valid)), valid.size)
    else:
        log.warning("Glebokos: 0 waznych pikseli (dysparycja>0: %d, odrzucone jako >%.0f mm: %d) - "
                    "sprawdz macierz Q i prog max_depth_mm", n_disp, max_depth_mm, too_far)
    # Duzy udzial pikseli obcietych progiem to sygnal zlej skali Q lub za niskiego progu.
    if n_disp and too_far > 0.5 * n_disp:
        log.warning("Glebokos: %d/%d pikseli (%.0f%%) przekracza max_depth=%.0f mm - "
                    "mozliwa zla skala macierzy Q lub za niski prog",
                    too_far, n_disp, 100 * too_far / n_disp, max_depth_mm)
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
        vals = d[mask]
        d_min, d_max = float(vals.min()), float(vals.max())
        if d_max > d_min:
            d[mask] = (vals - d_min) / (d_max - d_min) * 255.0
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
        vals = d[mask]
        d_min, d_max = float(vals.min()), float(vals.max())
        if d_max > d_min:
            d[mask] = (vals - d_min) / (d_max - d_min) * 255.0
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
    from logging_setup import setup_logging

    setup_logging()

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