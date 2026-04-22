# measurement.py
"""ETAP 6+7: Segmentacja obiektu nad paleta i pomiar jego wymiarow.

Pipeline:
  1. segment_object  - wyodrebnienie punktow obiektu powyzej powierzchni palety
  2. compute_bounding_box - obliczenie bounding box 3D
  3. extract_3d_contour   - obrys XY (convex hull)
  4. validate_measurement - ocena jakosci detekcji i pomiaru
  5. generate_report      - tekstowy raport po polsku

Uzycie:
    from pallet import detect_pallet
    from measurement import measure_object, validate_measurement, generate_report

    pallet_result = detect_pallet(xyz)
    meas = measure_object(xyz, pallet_result)
    validation = validate_measurement(meas)
    print(generate_report(meas, validation))
"""
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from pallet import PalletDetectionResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Struktury danych
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    width: float   # x_max - x_min [mm]
    length: float  # y_max - y_min [mm]
    height: float  # z_max - z_min [mm] - wysokosc nad paleta


@dataclass
class MeasurementResult:
    bbox: BoundingBox
    object_pts: np.ndarray      # (M,3) punkty obiektu w ukladzie palety
    contour_pts: np.ndarray     # (K,2) wierzcholki convex hull w XY
    pallet_result: PalletDetectionResult
    n_object_pts: int
    n_pallet_inliers: int


@dataclass
class ValidationReport:
    pallet_plane_rms_mm: float
    n_pallet_inliers: int
    pallet_coverage_ratio: float
    object_height_ok: bool
    object_within_roi: bool
    issues: list = field(default_factory=list)
    passed: bool = False


# ---------------------------------------------------------------------------
# Segmentacja obiektu
# ---------------------------------------------------------------------------

def segment_object(
    xyz_pallet: np.ndarray,
    noise_floor_mm: float = 20.0,
    max_height_mm: float = 2000.0,
) -> np.ndarray:
    """Zwraca maska bool punktow nalezacych do obiektu nad paleta.

    W ukladzie palety oS Z wskazuje ku kamerze: Z=0 to powierzchnia palety,
    Z > 0 to przestrzen nad paleta. Szum SGBM przy powierzchni daje Z ~ +-noise_floor,
    wiec odcinamy dolny prog.

    Args:
        xyz_pallet:    (N,3) chmura w ukladzie palety [mm]
        noise_floor_mm: minimalny prog wysokosci nad paleta [mm]
        max_height_mm:  maksymalny prog (filtr "latajacych" punktow)

    Returns:
        (N,) bool - True dla punktow obiektu
    """
    z = xyz_pallet[:, 2]
    return (z > noise_floor_mm) & (z < max_height_mm)


# ---------------------------------------------------------------------------
# Bounding box 3D
# ---------------------------------------------------------------------------

def compute_bounding_box(xyz_object: np.ndarray) -> BoundingBox:
    """Oblicza 3D bounding box chmury punktow obiektu.

    Args:
        xyz_object: (M,3) punkty obiektu w ukladzie palety [mm]

    Returns:
        BoundingBox z wymiarami w mm

    Raises:
        ValueError: jesli chmura jest pusta
    """
    if len(xyz_object) == 0:
        raise ValueError("Pusta chmura punktow - nie mozna wyliczyc bounding box")

    x_min, x_max = float(xyz_object[:, 0].min()), float(xyz_object[:, 0].max())
    y_min, y_max = float(xyz_object[:, 1].min()), float(xyz_object[:, 1].max())
    z_min, z_max = float(xyz_object[:, 2].min()), float(xyz_object[:, 2].max())

    return BoundingBox(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        width=x_max - x_min,
        length=y_max - y_min,
        height=z_max - z_min,
    )


# ---------------------------------------------------------------------------
# Obrys 3D (convex hull XY)
# ---------------------------------------------------------------------------

def extract_3d_contour(xyz_object: np.ndarray) -> np.ndarray:
    """Wyznacza obrys konturowy obiektu jako convex hull w plaszczyznie XY.

    Args:
        xyz_object: (M,3) punkty obiektu w ukladzie palety

    Returns:
        (K,2) wierzcholki wypuklego obrysu w ukladzie XY palety
    """
    pts_2d = xyz_object[:, :2].astype(np.float32)

    if len(pts_2d) < 3:
        return pts_2d

    hull = cv2.convexHull(pts_2d)
    return hull.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Glowna funkcja pomiaru
# ---------------------------------------------------------------------------

def measure_object(
    xyz: np.ndarray,
    pallet_result: PalletDetectionResult,
    noise_floor_mm: float = 20.0,
    max_height_mm: float = 2000.0,
) -> MeasurementResult:
    """Segmentuje obiekt nad paleta i mierzy jego wymiary 3D.

    Args:
        xyz:           (N,3) chmura punktow w ukladzie kamery [mm]
        pallet_result: wynik detekcji palety z detect_pallet()
        noise_floor_mm: min. wysokosc nad paleta [mm] (odcina szum powierzchni)
        max_height_mm:  max. wysokosc [mm]

    Returns:
        MeasurementResult z bbox, konturem i metadanymi

    Raises:
        RuntimeError: jesli brak punktow obiektu nad paleta
    """
    # Punkty ROI w ukladzie palety
    xyz_roi = pallet_result.xyz_pallet[pallet_result.roi_mask]

    obj_mask = segment_object(xyz_roi, noise_floor_mm, max_height_mm)

    if obj_mask.sum() == 0:
        raise RuntimeError(
            "Brak punktow obiektu nad paleta "
            f"(noise_floor={noise_floor_mm} mm, max={max_height_mm} mm)"
        )

    object_pts = xyz_roi[obj_mask]
    bbox = compute_bounding_box(object_pts)
    contour_pts = extract_3d_contour(object_pts)

    log.info(
        "Obiekt: %d pkt, W=%.0f L=%.0f H=%.0f mm",
        len(object_pts), bbox.width, bbox.length, bbox.height,
    )

    return MeasurementResult(
        bbox=bbox,
        object_pts=object_pts,
        contour_pts=contour_pts,
        pallet_result=pallet_result,
        n_object_pts=int(obj_mask.sum()),
        n_pallet_inliers=int(pallet_result.plane.inlier_mask.sum()),
    )


# ---------------------------------------------------------------------------
# Walidacja
# ---------------------------------------------------------------------------

def validate_measurement(
    result: MeasurementResult,
    pallet_width_mm: float = config.PALLET_WIDTH_MM,
    pallet_length_mm: float = config.PALLET_LENGTH_MM,
    min_height_mm: float = 10.0,
    max_height_mm: float = 2000.0,
    max_pallet_rms_mm: float = 30.0,
    min_inliers: int = 100,
) -> ValidationReport:
    """Ocenia jakosc detekcji palety i pomiaru obiektu.

    Sprawdzane kryteria:
      - RMS residual plaszczyzny < max_pallet_rms_mm
      - Liczba inlierow RANSAC >= min_inliers
      - Wysokosc obiektu w zakresie [min_height_mm, max_height_mm]
      - Obiekt nie wykracza poza obrys palety o wiecej niz 20%

    Returns:
        ValidationReport z lista problemow i ogolnym statusem PASS/FAIL
    """
    issues = []
    rms = result.pallet_result.plane.rms_residual
    n_inliers = result.n_pallet_inliers

    if rms > max_pallet_rms_mm:
        issues.append(f"Zly fit plaszczyzny: RMS={rms:.1f} mm > {max_pallet_rms_mm} mm")

    if n_inliers < min_inliers:
        issues.append(f"Za malo inlierow RANSAC: {n_inliers} < {min_inliers}")

    # Szacowanie pokrycia palety (rozstaw inlierow wzgledem wymiarow nominalnych)
    inlier_pallet = result.pallet_result.xyz_pallet[result.pallet_result.plane.inlier_mask]
    x_span = float(inlier_pallet[:, 0].max() - inlier_pallet[:, 0].min()) if len(inlier_pallet) > 1 else 0.0
    y_span = float(inlier_pallet[:, 1].max() - inlier_pallet[:, 1].min()) if len(inlier_pallet) > 1 else 0.0
    coverage = ((x_span / pallet_width_mm) + (y_span / pallet_length_mm)) / 2.0

    bbox = result.bbox
    object_height_ok = min_height_mm <= bbox.height <= max_height_mm
    if not object_height_ok:
        issues.append(f"Nieprawidlowa wysokosc obiektu: {bbox.height:.0f} mm")

    # Sprawdz czy obiekt nie wykracza znaczaco poza obrys ROI
    margin = 0.2  # 20% tolerancji
    half_w = pallet_width_mm / 2.0 * (1 + margin)
    half_l = pallet_length_mm / 2.0 * (1 + margin)
    object_within_roi = (
        abs(bbox.x_min) <= half_w and abs(bbox.x_max) <= half_w and
        abs(bbox.y_min) <= half_l and abs(bbox.y_max) <= half_l
    )
    if not object_within_roi:
        issues.append("Obiekt wykracza poza obrys palety (>20% tolerancji)")

    passed = len(issues) == 0

    return ValidationReport(
        pallet_plane_rms_mm=rms,
        n_pallet_inliers=n_inliers,
        pallet_coverage_ratio=coverage,
        object_height_ok=object_height_ok,
        object_within_roi=object_within_roi,
        issues=issues,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Raport tekstowy
# ---------------------------------------------------------------------------

def generate_report(result: MeasurementResult, validation: ValidationReport) -> str:
    """Generuje tekstowy raport pomiaru obiektu (po polsku).

    Returns:
        Wieloliniowy string gotowy do wyswietlenia lub zapisu do pliku
    """
    sep = "=" * 60
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    bbox = result.bbox

    lines = [
        "",
        sep,
        "  RAPORT POMIARU OBIEKTU NA EUROPALECIE",
        sep,
        f"  Data/czas: {ts}",
        "",
        "--- Detekcja plaszczyzny palety ---",
        f"  RMS residual:       {validation.pallet_plane_rms_mm:.2f} mm",
        f"  Inliery RANSAC:     {validation.n_pallet_inliers}",
        f"  Pokrycie palety:    {100*validation.pallet_coverage_ratio:.1f} %",
        "",
        "--- Wymiary obiektu (bounding box 3D) ---",
        f"  Szerokosc (X): {bbox.width:>8.1f} mm",
        f"  Dlugosc   (Y): {bbox.length:>8.1f} mm",
        f"  Wysokosc  (Z): {bbox.height:>8.1f} mm",
        f"  Liczba pkt.:   {result.n_object_pts}",
        "",
        "--- Walidacja ---",
    ]

    status = "PASS" if validation.passed else "FAIL"
    lines.append(f"  Status: {status}")

    if validation.issues:
        lines.append("  Problemy:")
        for issue in validation.issues:
            lines.append(f"    - {issue}")
    else:
        lines.append("  Brak problemow - pomiar prawidlowy")

    lines.append(sep)

    return "\n".join(lines)
