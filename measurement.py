# measurement.py
"""Etap 6+7 pipeline: segmentacja obiektu nad paleta i pomiar jego wymiarow 3D.

Dane wejsciowe: PalletDetectionResult z pallet.py (chmura w ukladzie palety).
Dane wyjsciowe: wymiary bounding box (szerokosc, dlugosc, wysokosc) w mm.

Kolejnosc przetwarzania:
  1. segment_object()      - wyodrebnia punkty powyzej powierzchni palety (Z > noise_floor)
  2. compute_bounding_box() - oblicza prostopadloscienny bbox 3D na tych punktach
  3. extract_3d_contour()   - wyznacza obrys XY jako convex hull (do wizualizacji)
  4. validate_measurement() - sprawdza kryteria jakosci pomiaru i detekcji palety
  5. generate_report()      - formatuje wyniki do czytelnego raportu tekstowego

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
    """Prostopadloscienny bounding box 3D obiektu w ukladzie palety.

    Wszystkie wartosci w milimetrach. Osie zgodne z ukladem palety:
      X - szerokosc (rownolega do krotszej krawedzi palety)
      Y - dlugosc   (rownolega do dluzszej krawedzi palety)
      Z - wysokosc  (prostopadla do powierzchni palety, ku kamerze)
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    width: float   # x_max - x_min [mm]
    length: float  # y_max - y_min [mm]
    height: float  # z_max - z_min [mm] - wysokosc nad powierzchnia palety


@dataclass
class VolumeEstimate:
    """Trzy komplementarne szacunki objetosci obiektu (wszystko w mm^3).

    Stereo widzi tylko przednia/gorna powierzchnie obiektu (tyl jest zaslonity),
    dlatego liczymy trzy szacunki o roznych zalozeniach:
      - voxel_mm3: bryla pelna od palety do obserwowanej gory (najbardziej realistyczny)
      - bbox_mm3:  W*L*H prostopadloscianu (gorne ograniczenie)
      - hull_mm3:  objetosc otoczki wypuklej widocznych punktow (zwykle niedoszacowanie)
    """
    bbox_mm3: float
    voxel_mm3: float
    hull_mm3: float | None       # None gdy scipy niedostepne lub chmura zdegenerowana
    voxel_cell_mm: float         # rozmiar komorki uzyty do integracji height-field
    n_voxel_cells: int           # liczba zajetych komorek siatki XY
    fill_ratio: float            # voxel_mm3 / bbox_mm3 - "pelnosc" bryly w [0..1]


@dataclass
class MeasurementResult:
    """Wynik pomiaru obiektu - bbox, objetosc, punkty i metadane do walidacji."""
    bbox: BoundingBox
    volume: VolumeEstimate      # szacunki objetosci obiektu
    object_pts: np.ndarray      # (M,3) punkty obiektu w ukladzie palety [mm]
    contour_pts: np.ndarray     # (K,2) wierzcholki convex hull w plaszczyznie XY
    pallet_result: PalletDetectionResult  # oryginalny wynik detekcji palety
    n_object_pts: int           # liczba punktow zaklasyfikowanych jako obiekt
    n_pallet_inliers: int       # liczba inlierow RANSAC palety (miara jakosci fit)


@dataclass
class ValidationReport:
    """Raport walidacyjny pomiaru - lista wykrytych problemow i ogolny status.

    passed=True oznacza, ze wszystkie kryteria jakosci zostaly spelnione.
    Jesli passed=False, lista issues zawiera opisy konkretnych problemow.
    """
    pallet_plane_rms_mm: float    # RMS residual plaszczyzny palety [mm]
    n_pallet_inliers: int         # liczba inlierow RANSAC
    pallet_coverage_ratio: float  # stosunek pokrycia palety przez inliery [0..1]
    object_height_ok: bool        # czy wysokosc obiektu w sensownym zakresie
    object_within_roi: bool       # czy obiekt nie wykracza poza obrys palety
    issues: list = field(default_factory=list)  # lista opisow wykrytych problemow
    passed: bool = False          # True jesli brak problemow


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
    Z > 0 to przestrzen nad paleta. Szum SGBM przy powierzchni daje Z  +-noise_floor,
    wiec odcinamy dolny prog.

    Args:
        xyz_pallet:    (N,3) chmura w ukladzie palety [mm]
        noise_floor_mm: minimalny prog wysokosci nad paleta [mm]
        max_height_mm:  maksymalny prog (filtr "latajacych" punktow)

    Returns:
        (N,) bool - True dla punktow obiektu
    """
    z = xyz_pallet[:, 2]
    mask = (z > noise_floor_mm) & (z < max_height_mm)
    # Rozbijamy odrzucone punkty na "ponizej progu szumu" i "ponad limitem wysokosci",
    # zeby od razu bylo widac, czy obiekt zniknal przez za wysoki noise_floor,
    # czy przez latajace punkty ponad max_height.
    below = int((z <= noise_floor_mm).sum())
    above = int((z >= max_height_mm).sum())
    log.debug("Segmentacja: %d/%d pkt jako obiekt (ponizej %.0f mm: %d, ponad %.0f mm: %d)",
              int(mask.sum()), len(z), noise_floor_mm, below, max_height_mm, above)
    return mask


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

    # Wymiar bliski 0 oznacza zwykle, ze obiekt zredukowal sie do cienkiej warstwy
    # punktow (zla segmentacja) - sygnalizujemy, bo objetosc bedzie wtedy bezuzyteczna.
    w, l, h = x_max - x_min, y_max - y_min, z_max - z_min
    if min(w, l, h) < 1.0:
        log.warning("Bounding box ma znikomy wymiar: W=%.1f L=%.1f H=%.1f mm - "
                    "prawdopodobnie zla segmentacja obiektu", w, l, h)

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

    Rzutuje punkty obiektu na plaszczyzne XY (ignorujac wysokosc Z)
    i oblicza wypukla otoczke. Wynik nadaje sie do wizualizacji rzutu
    z gory oraz do przyblizonych obliczen powierzchni podstawy obiektu.

    Args:
        xyz_object: (M,3) punkty obiektu w ukladzie palety [mm]

    Returns:
        (K,2) wierzcholki wypuklego obrysu w plaszczyznie XY [mm]
    """
    pts_2d = xyz_object[:, :2].astype(np.float32)

    if len(pts_2d) < 3:
        # Za malo punktow na convex hull - zwracamy je bez zmian
        return pts_2d

    hull = cv2.convexHull(pts_2d)
    return hull.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Estymacja objetosci
# ---------------------------------------------------------------------------

def compute_voxel_volume(
    xyz_object: np.ndarray,
    cell_mm: float = 10.0,
) -> tuple[float, int]:
    """Objetosc metoda kolumnowa (height-field) - calka wysokosci po footprintcie XY.

    Dzielimy rzut obiektu na plaszczyzne XY na kwadratowe komorki cell_mm x cell_mm.
    Dla kazdej zajetej komorki bierzemy max(Z) punktow w niej, co reprezentuje
    gorna powierzchnie obiektu, i mnozymy przez pole komorki. Suma to objetosc
    bryly pelnej od powierzchni palety (Z=0) do obserwowanej gory.

    Uwaga: calkujemy od Z=0 (paleta), a nie od z_min obiektu. BoundingBox.height
    liczy z_max - z_min (z_min ~ noise_floor 20-25 mm), wiec objetosc kolumnowa
    swiadomie obejmuje rowniez warstwe przy palecie - fizycznie poprawne dla
    obiektu stojacego na palecie.

    Args:
        xyz_object: (M,3) punkty obiektu w ukladzie palety [mm], Z = wysokosc nad paleta
        cell_mm:    rozmiar komorki siatki XY [mm]

    Returns:
        (volume_mm3, n_zajetych_komorek)
    """
    if len(xyz_object) == 0:
        return 0.0, 0

    x = xyz_object[:, 0]
    y = xyz_object[:, 1]
    z = np.clip(xyz_object[:, 2], 0.0, None)  # wysokosc liczona od palety, nie ujemna

    ix = np.floor((x - x.min()) / cell_mm).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_mm).astype(np.int64)
    n_y = int(iy.max()) + 1
    flat = ix * n_y + iy  # plaski indeks komorki

    # Dla kazdej komorki max(Z): np.maximum.at akumuluje maksimum po powtarzajacych
    # sie indeksach (rozne punkty wpadajace do tej samej komorki).
    cell_max = np.zeros(int(flat.max()) + 1, dtype=np.float64)
    np.maximum.at(cell_max, flat, z)

    occupied = np.unique(flat)
    volume_mm3 = float(cell_max[occupied].sum() * cell_mm * cell_mm)
    return volume_mm3, int(len(occupied))


def compute_hull_volume(xyz_object: np.ndarray) -> float | None:
    """Objetosc otoczki wypuklej 3D punktow obiektu (scipy.spatial.ConvexHull).

    Zwraca objetosc geometrycznej otoczki widocznych punktow. Poniewaz stereo
    rejestruje tylko przednia/gorna powierzchnie, otoczka zwykle niedoszacowuje
    rzeczywistej objetosci (brak tylnej sciany).

    Returns:
        objetosc [mm^3] lub None gdy scipy niedostepne, za malo punktow
        (<4) albo chmura zdegenerowana (wspolplaszczyznowa - QhullError).
    """
    if len(xyz_object) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(xyz_object).volume)
    except ImportError:
        log.warning("scipy niedostepne - pomijam objetosc convex hull")
        return None
    except Exception as e:
        # QhullError przy zdegenerowanej (wspolplaszczyznowej) chmurze
        log.warning("Nie udalo sie policzyc convex hull: %s", e)
        return None


def estimate_volume(
    xyz_object: np.ndarray,
    bbox: BoundingBox,
    cell_mm: float = 10.0,
) -> VolumeEstimate:
    """Liczy trzy szacunki objetosci obiektu: voxel-column, bbox i convex hull.

    Args:
        xyz_object: (M,3) punkty obiektu w ukladzie palety [mm]
        bbox:       bounding box obiektu z compute_bounding_box()
        cell_mm:    rozmiar komorki dla metody kolumnowej [mm]

    Returns:
        VolumeEstimate z trzema szacunkami i wskaznikiem fill_ratio
    """
    bbox_mm3 = bbox.width * bbox.length * bbox.height
    voxel_mm3, n_cells = compute_voxel_volume(xyz_object, cell_mm)
    hull_mm3 = compute_hull_volume(xyz_object)
    fill_ratio = float(voxel_mm3 / bbox_mm3) if bbox_mm3 > 0 else 0.0

    log.info(
        "Objetosc: voxel=%.0f mm^3 bbox=%.0f mm^3 hull=%s fill=%.0f%%",
        voxel_mm3, bbox_mm3,
        f"{hull_mm3:.0f} mm^3" if hull_mm3 is not None else "n/d",
        100 * fill_ratio,
    )

    return VolumeEstimate(
        bbox_mm3=bbox_mm3,
        voxel_mm3=voxel_mm3,
        hull_mm3=hull_mm3,
        voxel_cell_mm=cell_mm,
        n_voxel_cells=n_cells,
        fill_ratio=fill_ratio,
    )


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
    # Bierzemy tylko punkty wewnatrz ROI palety (juz w ukladzie palety)
    xyz_roi = pallet_result.xyz_pallet[pallet_result.roi_mask]
    log.debug("measure_object: %d punktow w ROI palety (z %d w chmurze)",
              len(xyz_roi), len(pallet_result.xyz_pallet))

    # Segmentujemy obiekt: punkty powyzej noise_floor to obiekt, reszta to szum lub paleta
    obj_mask = segment_object(xyz_roi, noise_floor_mm, max_height_mm)

    if obj_mask.sum() == 0:
        raise RuntimeError(
            "Brak punktow obiektu nad paleta "
            f"(noise_floor={noise_floor_mm} mm, max={max_height_mm} mm)"
        )

    object_pts = xyz_roi[obj_mask]
    # Mala liczba punktow daje niestabilny bbox i objetosc - ostrzegamy, ale liczymy dalej.
    if len(object_pts) < 50:
        log.warning("Obiekt ma tylko %d punktow - wymiary i objetosc moga byc niedokladne",
                    len(object_pts))
    bbox = compute_bounding_box(object_pts)
    volume = estimate_volume(object_pts, bbox)
    contour_pts = extract_3d_contour(object_pts)

    log.info(
        "Obiekt: %d pkt, W=%.0f L=%.0f H=%.0f mm",
        len(object_pts), bbox.width, bbox.length, bbox.height,
    )

    return MeasurementResult(
        bbox=bbox,
        volume=volume,
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

    # Szacowanie pokrycia palety: sprawdzamy, jaka czesc nominalnej powierzchni
    # palety jest objeta przez inliery RANSAC. Niska wartość (<0.5) sugeruje,
    # ze paleta jest czesciowo poza kadrem lub zaslonicta.
    inlier_pallet = result.pallet_result.xyz_pallet[result.pallet_result.plane.inlier_mask]
    x_span = float(inlier_pallet[:, 0].max() - inlier_pallet[:, 0].min()) if len(inlier_pallet) > 1 else 0.0
    y_span = float(inlier_pallet[:, 1].max() - inlier_pallet[:, 1].min()) if len(inlier_pallet) > 1 else 0.0
    coverage = ((x_span / pallet_width_mm) + (y_span / pallet_length_mm)) / 2.0

    bbox = result.bbox
    object_height_ok = min_height_mm <= bbox.height <= max_height_mm
    if not object_height_ok:
        issues.append(f"Nieprawidlowa wysokosc obiektu: {bbox.height:.0f} mm")

    # Tolerancja 20% pozwala na drobne bledy transformacji do ukladu palety
    # i niewielkie wysuniecie obiektu poza krawedz. Przekroczenie sugeruje
    # blad detekcji plaszczyzny lub zle ustawienie kamer.
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

    # Logujemy wynik walidacji - kazdy problem jako osobne ostrzezenie ulatwia
    # zlokalizowanie przyczyny w logach bez czytania calego raportu tekstowego.
    if passed:
        log.info("Walidacja pomiaru: PASS (RMS=%.1f mm, inliery=%d, pokrycie=%.0f%%)",
                 rms, n_inliers, 100 * coverage)
    else:
        log.warning("Walidacja pomiaru: FAIL - %d problem(ow):", len(issues))
        for issue in issues:
            log.warning("  - %s", issue)

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
    """Generuje tekstowy raport pomiaru obiektu gotowy do zapisu lub wyswietlenia.

    Raport zawiera: timestamp, statystyki detekcji palety, wymiary obiektu
    i wynik walidacji z lista ewentualnych problemow.

    Returns:
        Wieloliniowy string zakonczony separatorem - gotowy do print() lub zapisu do pliku
    """
    sep = "=" * 60
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    bbox = result.bbox
    vol = result.volume

    # 1 litr (dm^3) = 1e6 mm^3
    hull_line = (
        f"  Convex hull:   {vol.hull_mm3:>12.0f} mm^3 = {vol.hull_mm3/1e6:6.2f} l"
        if vol.hull_mm3 is not None
        else "  Convex hull:   niedostepna (scipy brak / chmura zdegenerowana)"
    )

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
        "--- Objetosc obiektu ---",
        f"  Voxel-column:  {vol.voxel_mm3:>12.0f} mm^3 = {vol.voxel_mm3/1e6:6.2f} l  (komorka {vol.voxel_cell_mm:.0f} mm, {vol.n_voxel_cells} kom.)",
        f"  Bounding box:  {vol.bbox_mm3:>12.0f} mm^3 = {vol.bbox_mm3/1e6:6.2f} l",
        hull_line,
        f"  Wsk. pelnosci: {100*vol.fill_ratio:.1f} % (voxel / bbox)",
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