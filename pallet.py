# pallet.py
"""Etap 5 pipeline: detekcja plaszczyzny europalety w chmurze punktow 3D.

Algorytm sklada sie z czterech krokow:
  1. RANSAC         - losuje trojki punktow, szuka plaszczyzny z max. liczba inlierow
  2. SVD-refinement - dokladna normalna z dekompozycji SVD na wszystkich inlierach
  3. Transformacja  - obraca chmure tak, by powierzchnia palety lezala w Z=0
  4. Filtr ROI      - zachowuje tylko punkty wewnatrz gabarytu 1200x800 mm

Rezultatem jest PalletDetectionResult z chmura w ukladzie palety i maska ROI,
ktore sa nastepnie przekazywane do measurement.py (etap 6+7).

Uzycie:
    from pallet import detect_pallet, PalletDetectionResult
    result = detect_pallet(xyz)
    # result.xyz_pallet - chmura w ukladzie palety (Z=0 to powierzchnia)
    # result.roi_mask   - maska bool punktow wewnatrz obrysu palety
"""
import logging
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Struktury danych
# ---------------------------------------------------------------------------

@dataclass
class PlaneModel:
    """Wyznaczona plaszczyzna z parametrami i statystykami dopasowania.

    Rownanie plaszczyzny: normal . x + d = 0
    Odleglosc punktu p od plaszczyzny: |normal . p + d|
    """
    normal: np.ndarray       # (3,) znormalizowany wektor normalny, wskazuje w strone kamery
    d: float                 # wyraz wolny rownania plaszczyzny
    inlier_mask: np.ndarray  # (N,) bool - maska punktow lezacych na plaszczyznie
    inlier_pts: np.ndarray   # (M,3) - wspolrzedne inlierow [mm]
    rms_residual: float      # RMS odleglosci inlierow od plaszczyzny [mm]; im mniejszy, tym lepszy fit


@dataclass
class PalletDetectionResult:
    """Wynik detekcji europalety - zawiera chmure w ukladzie palety i maska ROI.

    Po detekcji chmura jest juz przeksztalcona do ukladu palety (Z=0 = powierzchnia),
    co pozwala bezposrednio segmentowac obiekty lezace powyzej (Z > 0).
    """
    plane: PlaneModel
    xyz_pallet: np.ndarray   # (N,3) cala chmura w ukladzie palety [mm]
    roi_mask: np.ndarray     # (N,) bool - True dla punktow wewnatrz gabarytu 1200x800 mm
    R: np.ndarray            # (3,3) macierz obrotu: uklad kamery -> uklad palety
    centroid: np.ndarray     # (3,) srodek inlierow w ukladzie kamery (origin ukladu palety)


# ---------------------------------------------------------------------------
# RANSAC - detekcja dominujacej plaszczyzny
# ---------------------------------------------------------------------------

def detect_pallet_plane(
    xyz: np.ndarray,
    n_iterations: int = 1000,
    distance_threshold: float = 10.0,
    min_inliers: int = 50,
    rng_seed: int = 0,
) -> PlaneModel:
    """Wykrywa dominujaca plaszczyzne (paleta) metodą RANSAC.

    Args:
        xyz:                (N,3) chmura punktow [mm]
        n_iterations:       liczba iteracji RANSAC
        distance_threshold: prog odleglosci punktu od plaszczyzny [mm]
        min_inliers:        minimalna liczba inlierow, ponizej rzuca RuntimeError
        rng_seed:           ziarno generatora dla powtarzalnosci

    Returns:
        PlaneModel z normalną wskazujacą w stronę kamery (origin)

    Raises:
        ValueError:   jesli mniej niz 3 punkty w chmurze
        RuntimeError: jesli nie znaleziono plaszczyzny z wystarczajaca liczba inlierow
    """
    if len(xyz) < 3:
        raise ValueError(f"Za malo punktow do RANSAC: {len(xyz)} < 3")

    rng = np.random.RandomState(rng_seed)
    n = len(xyz)
    log.debug("RANSAC start: %d punktow, %d iteracji, prog=%.1f mm, min_inliers=%d",
              n, n_iterations, distance_threshold, min_inliers)

    best_inlier_count = 0
    best_normal = None
    best_d = None
    n_degenerate = 0  # iteracje odrzucone z powodu wspoliniowych punktow

    for _ in range(n_iterations):
        # Losujemy 3 rozne punkty - minimalna liczba do wyznaczenia plaszczyzny
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]

        # Normalna plaszczyzny = iloczyn wektorowy dwoch krawedzi trojkata
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            # Punkty sa wspoliniowe - pomijamy ta iteracje
            n_degenerate += 1
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p0)

        # Liczymy inlierów: punkty odlegle od plaszczyzny o mniej niz distance_threshold
        dists = np.abs(xyz @ normal + d)
        inlier_count = int((dists < distance_threshold).sum())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    # Duzy udzial zdegenerowanych iteracji sugeruje chmure prawie wspoliniowa/rzadka.
    if n_degenerate > 0.5 * n_iterations:
        log.warning("RANSAC: %d/%d iteracji zdegenerowanych (wspoliniowe punkty) - "
                    "chmura moze byc zbyt rzadka lub plaska", n_degenerate, n_iterations)

    if best_inlier_count < min_inliers:
        raise RuntimeError(
            f"Nie znaleziono plaszczyzny palety: max inlierow={best_inlier_count} < {min_inliers}"
        )

    assert best_normal is not None and best_d is not None

    # SVD-refinement: RANSAC daje przyblizona plaszczyzne, SVD na inlierach
    # wyznacza dokladna normalna metoda minimalnej wariancji (PCA).
    # Ostatni wiersz Vt odpowiada kierunkowi najmniejszej zmiennosci - czyli
    # prostopadlemu do plaszczyzny, czyli normalnej.
    inlier_mask_rough = np.abs(xyz @ best_normal + best_d) < distance_threshold
    inlier_pts = xyz[inlier_mask_rough]

    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal_refined = Vt[-1]  # ostatni wiersz = normalna (min wariancja)
    normal_refined = normal_refined / np.linalg.norm(normal_refined)

    # Konwencja znaku: normalna musi wskazywac w strone kamery (origin),
    # bo to ona bedzie podstawa ukladu wspolrzednych palety (os Z ku gorze).
    # Centroid plaszczyzny jest "za" kamera, wiec dot(normal, centroid) < 0
    # oznacza, ze normalna wskazuje ku kamerze.
    # Jesli dot > 0, obracamy kierunek normalnej.
    if np.dot(normal_refined, centroid) > 0:
        normal_refined = -normal_refined

    d_refined = -np.dot(normal_refined, centroid)

    # Ponowna maska inlierow z uzyciem udokadnionej normalnej po SVD
    residuals = xyz @ normal_refined + d_refined
    inlier_mask = np.abs(residuals) < distance_threshold
    inlier_pts_final = xyz[inlier_mask]
    rms = float(np.sqrt((residuals[inlier_mask] ** 2).mean()))

    inlier_ratio = inlier_mask.mean()
    log.info(
        "RANSAC: znaleziono plaszczyzne, %d inlierow (%.1f%%), RMS=%.2f mm, normalna=[%.2f %.2f %.2f]",
        inlier_mask.sum(), 100 * inlier_ratio, rms, *normal_refined,
    )
    # Wysoki RMS = punkty slabo leza na plaszczyznie (zaszumiona chmura lub
    # paleta nie jest dominujaca plaszczyzna). Prog 30 mm jak w validate_measurement.
    if rms > 30.0:
        log.warning("RANSAC: wysoki RMS=%.1f mm - dopasowanie plaszczyzny slabe "
                    "(zaszumiona chmura lub bledna detekcja palety)", rms)
    # Niski udzial inlierow = plaszczyzna palety nie dominuje w scenie.
    if inlier_ratio < 0.2:
        log.warning("RANSAC: tylko %.0f%% punktow lezy na plaszczyznie - paleta moze nie byc "
                    "dominujaca plaszczyzna (sprawdz kadr)", 100 * inlier_ratio)

    return PlaneModel(
        normal=normal_refined,
        d=d_refined,
        inlier_mask=inlier_mask,
        inlier_pts=inlier_pts_final,
        rms_residual=rms,
    )


# ---------------------------------------------------------------------------
# Transformacja do ukladu wspolrzednych palety
# ---------------------------------------------------------------------------

def transform_to_pallet_frame(
    xyz: np.ndarray,
    plane: PlaneModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transformuje chmure do ukladu wspolrzednych palety.

    Uklad palety:
      - origin = centroid inlierow RANSAC
      - os Z  = plane.normal (ku kamerze = dodatnia wysokosc nad paleta)
      - os X  = prostopadla do Z, mozliwie rownolega do [1,0,0]
      - os Y  = X x Z (prawoskretny uklad)

    Returns:
        (xyz_pallet, R, centroid)
        - xyz_pallet: (N,3) wspolrzedne w ukladzie palety
        - R:          (3,3) macierz obrotu (wiersze = wektory bazowe)
        - centroid:   (3,) srodek ciezkosc inlierow w ukladzie kamery
    """
    centroid = plane.inlier_pts.mean(axis=0)
    z_axis = plane.normal  # juz znormalizowany, wskazuje ku kamerze

    # Wybor wektora referencyjnego do budowy osi X ukladu palety.
    # Jesli normalna jest zbyt rownolega do [1,0,0], uzywamy [0,1,0]
    # aby uniknac degeneracji przy iloczynie wektorowym.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    # Gram-Schmidt: rzutujemy ref na plaszczyzne prostopadla do Z, normalizujemy
    x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)  # trzecia os - prawoskretny uklad

    R = np.vstack([x_axis, y_axis, z_axis])  # wiersze = wektory bazowe
    # Rotacja i translacja: przenosimy cala chmure do ukladu palety
    xyz_pallet = (R @ (xyz - centroid).T).T

    log.debug("Uklad palety: centroid=[%.0f %.0f %.0f] mm, zakres Z po transformacji [%.0f, %.0f] mm",
              *centroid, float(xyz_pallet[:, 2].min()), float(xyz_pallet[:, 2].max()))
    return xyz_pallet, R, centroid


def filter_roi(
    xyz_pallet: np.ndarray,
    pallet_width_mm: float = config.PALLET_WIDTH_MM,
    pallet_length_mm: float = config.PALLET_LENGTH_MM,
) -> np.ndarray:
    """Zwraca maske bool punktow lezacych wewnatrz obrysu europalety.

    Zaklada, ze origin ukladu palety pokrywa sie ze srodkiem cizkosci
    inlierow RANSAC (czyli mniej wiecej srodkiem palety). Punkty spoza
    gabarytu nominalnego sa odrzucane - nie beda uzyte do pomiaru obiektu.

    Args:
        xyz_pallet:      (N,3) chmura w ukladzie palety [mm]
        pallet_width_mm: nominalna szerokosc palety (os X) [mm]
        pallet_length_mm: nominalna dlugosc palety (os Y) [mm]

    Returns:
        (N,) bool - True dla punktow wewnatrz gabarytu
    """
    half_w = pallet_width_mm / 2.0
    half_l = pallet_length_mm / 2.0
    mask = (
        (np.abs(xyz_pallet[:, 0]) <= half_w) &
        (np.abs(xyz_pallet[:, 1]) <= half_l)
    )
    log.info(
        "ROI %dx%d mm: %d/%d punktow wewnatrz (%.1f%%)",
        int(pallet_width_mm), int(pallet_length_mm),
        mask.sum(), len(xyz_pallet), 100 * mask.mean(),
    )
    if mask.sum() == 0:
        log.warning("ROI: 0 punktow wewnatrz obrysu palety - origin ukladu palety moze byc "
                    "przesuniety lub gabaryt %dx%d mm zle dobrany",
                    int(pallet_width_mm), int(pallet_length_mm))
    return mask


# ---------------------------------------------------------------------------
# Glowna funkcja detekcji - wrapper
# ---------------------------------------------------------------------------

def detect_pallet(
    xyz: np.ndarray,
    n_iterations: int = 1000,
    distance_threshold: float = 10.0,
    min_inliers: int = 50,
    pallet_width_mm: float = config.PALLET_WIDTH_MM,
    pallet_length_mm: float = config.PALLET_LENGTH_MM,
    rng_seed: int = 0,
) -> PalletDetectionResult:
    """Kompleksowa detekcja europalety w chmurze punktow.

    Kolejne kroki:
      1. detect_pallet_plane (RANSAC + SVD)
      2. transform_to_pallet_frame
      3. filter_roi

    Args:
        xyz:               (N,3) chmura punktow w ukladzie kamery [mm]
        n_iterations:      iteracje RANSAC
        distance_threshold: prog [mm] dla inlierow
        min_inliers:       minimalna liczba inlierow
        pallet_width_mm:   szerokosc palety [mm] (domyslnie z config)
        pallet_length_mm:  dlugosc palety [mm] (domyslnie z config)
        rng_seed:          ziarno RNG dla powtarzalnosci

    Returns:
        PalletDetectionResult

    Raises:
        ValueError:   jesli za malo punktow
        RuntimeError: jesli brak dominujacej plaszczyzny
    """
    plane = detect_pallet_plane(xyz, n_iterations, distance_threshold, min_inliers, rng_seed)
    xyz_pallet, R, centroid = transform_to_pallet_frame(xyz, plane)
    roi_mask = filter_roi(xyz_pallet, pallet_width_mm, pallet_length_mm)

    # rng_seed=0 zapewnia powtarzalnosc wynikow przy tych samych danych wejsciowych
    return PalletDetectionResult(
        plane=plane,
        xyz_pallet=xyz_pallet,
        roi_mask=roi_mask,
        R=R,
        centroid=centroid,
    )


# ---------------------------------------------------------------------------
# CLI - szybki test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from logging_setup import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Test detekcji plaszczyzny palety")
    parser.add_argument("--cloud", required=True, help="Plik .npy z chmura punktow (N,3)")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()

    xyz_data = np.load(args.cloud)
    result = detect_pallet(xyz_data, n_iterations=args.iterations,
                           distance_threshold=args.threshold)

    print(f"\nNormalna plaszczyzny: {result.plane.normal}")
    print(f"RMS residual:         {result.plane.rms_residual:.2f} mm")
    print(f"Liczba inlierow:      {result.plane.inlier_mask.sum()}")
    print(f"Punkty w ROI:         {result.roi_mask.sum()}")