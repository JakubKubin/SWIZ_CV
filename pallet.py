# pallet.py
"""ETAP 5: Detekcja plaszczyzny europalety w chmurze punktow 3D.

Algorytm:
  1. RANSAC - losowe 3 punkty -> plaszczyzna -> liczymy inliery
  2. SVD-refinement na inlierach - dokladna normalna
  3. Transformacja do ukladu wspolrzednych palety (Z=0 to powierzchnia)
  4. Filtr ROI - punkty wewnatrz 1200x800 mm

Uzycie:
    from pallet import detect_pallet, PalletDetectionResult
    result = detect_pallet(xyz)
    # result.xyz_pallet - chmura w ukladzie palety
    # result.roi_mask   - punkty wewnatrz obrysu palety
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
    normal: np.ndarray       # (3,) znormalizowany wektor normalny, wskazuje w strone kamery
    d: float                 # wspolczynnik: normal . x + d = 0
    inlier_mask: np.ndarray  # (N,) bool - maska inlierow
    inlier_pts: np.ndarray   # (M,3) - punkty nalezace do plaszczyzny
    rms_residual: float      # RMS odleglosci inlierow od plaszczyzny [mm]


@dataclass
class PalletDetectionResult:
    plane: PlaneModel
    xyz_pallet: np.ndarray   # (N,3) cala chmura w ukladzie palety
    roi_mask: np.ndarray     # (N,) bool - punkty wewnatrz 1200x800 mm
    R: np.ndarray            # (3,3) macierz obrotu: uklad kamery -> uklad palety
    centroid: np.ndarray     # (3,) centroid inlierow w ukladzie kamery


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

    best_inlier_count = 0
    best_normal = None
    best_d = None

    for _ in range(n_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]

        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p0)

        dists = np.abs(xyz @ normal + d)
        inlier_count = int((dists < distance_threshold).sum())

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_inlier_count < min_inliers:
        raise RuntimeError(
            f"Nie znaleziono plaszczyzny palety: max inlierow={best_inlier_count} < {min_inliers}"
        )

    # Refinement SVD na inlierach
    inlier_mask_rough = np.abs(xyz @ best_normal + best_d) < distance_threshold
    inlier_pts = xyz[inlier_mask_rough]

    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal_refined = Vt[-1]  # ostatni wiersz = normalna (min wariancja)
    normal_refined = normal_refined / np.linalg.norm(normal_refined)

    # Konwencja znaku: normalna musi wskazywac w strone kamery (origin)
    # Centroid plaszczyzny jest "za" kamera, wiec dot(normal, centroid) < 0
    # oznacza, ze normalna wskazuje ku kamerze - to jest poprawne.
    # Jesli dot > 0, obracamy.
    if np.dot(normal_refined, centroid) > 0:
        normal_refined = -normal_refined

    d_refined = -np.dot(normal_refined, centroid)

    # Finalna maska inlierow po SVD
    residuals = xyz @ normal_refined + d_refined
    inlier_mask = np.abs(residuals) < distance_threshold
    inlier_pts_final = xyz[inlier_mask]
    rms = float(np.sqrt((residuals[inlier_mask] ** 2).mean()))

    log.info(
        "RANSAC: znaleziono plaszczyzne, %d inlierow (%.1f%%), RMS=%.2f mm",
        inlier_mask.sum(), 100 * inlier_mask.mean(), rms,
    )

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
    z_axis = plane.normal  # juz znormalizowany

    # Wybor referencyjnego wektora X
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.vstack([x_axis, y_axis, z_axis])  # wiersze = wektory bazowe
    xyz_pallet = (R @ (xyz - centroid).T).T

    return xyz_pallet, R, centroid


def filter_roi(
    xyz_pallet: np.ndarray,
    pallet_width_mm: float = config.PALLET_WIDTH_MM,
    pallet_length_mm: float = config.PALLET_LENGTH_MM,
) -> np.ndarray:
    """Zwraca maska bool punktow wewnatrz obrysu palety (wg wymirow nominalnych).

    Zaklada, ze origin ukladu palety = centroid plaszczyzny = srodek palety.
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
