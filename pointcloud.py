# pointcloud.py
"""Generowanie, filtrowanie i zapis chmury punktow 3D z mapy dysparycji.

Przepływ:
  1. build_pointcloud()   - reprojekcja dysparycji -> wspolrzedne XYZ [mm] + kolory RGB
  2. filter_pointcloud()  - usuniecie szumowych "latajacych" punktow (filtr statystyczny)
  3. save_ply()           - zapis do pliku PLY (ASCII lub binarny)
  4. render_topdown/sideview() - szybki rzut 2D do weryfikacji bez narzedzi 3D

Uzycie CLI:
    python pointcloud.py --calib calib_output/stereo.json \\
                         --disparity depth_output/disparity.npy \\
                         --color left_rect.png \\
                         --output depth_output/cloud.ply

Z kodu:
    from pointcloud import build_pointcloud, save_ply, filter_pointcloud
    xyz, colors = build_pointcloud(disparity, stereo.Q, left_rect)
    xyz, colors = filter_pointcloud(xyz, colors)
    save_ply("cloud.ply", xyz, colors)
"""
import logging
import struct
import numpy as np
import cv2
from pathlib import Path

from calibration import StereoParams, load_params
import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budowanie chmury punktow
# ---------------------------------------------------------------------------

def build_pointcloud(
    disparity: np.ndarray,
    Q: np.ndarray,
    color_image: np.ndarray | None = None,
    max_depth_mm: float = config.MAX_DEPTH_MM,
    min_depth_mm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Buduje chmure punktow 3D z mapy dysparycji i macierzy Q.

    Macierz Q pochodzi z StereoParams.Q wyznaczonej podczas kalibracji stereo
    przez cv2.stereoRectify. cv2.reprojectImageTo3D mnozy kazdy piksel
    [x, y, d, 1]^T przez Q i zwraca wspolrzedne [X, Y, Z] w mm
    (jesli kalibracja byla prowadzona w milimetrach).

    Args:
        disparity:    mapa dysparycji float32 z compute_disparity()
        Q:            stereo.Q - macierz 4x4 z StereoParams
        color_image:  opcjonalny obraz BGR do kolorowania punktow (np. left_rect)
        max_depth_mm: odfiltrowuje punkty dalej niz ten prog
        min_depth_mm: odfiltrowuje punkty blizej (szum przy krawedzi)

    Returns:
        xyz:    (N, 3) float32 - wspolrzedne punktow w mm
        colors: (N, 3) uint8 RGB lub None jesli brak color_image
    """
    points_3d = cv2.reprojectImageTo3D(disparity, Q)  # (H, W, 3) -> X, Y, Z [mm]

    # Maska waznosci: odrzucamy piksele bez dysparycji, z nieskonczonymi
    # wspolrzednymi (efekt dzielenia przez zero przy d=0) i spoza zakresu
    # pomiarowego (za bliskie = szum krawedzi, za dalekie = nierzetelne dane)
    mask = (
        (disparity > 0) &
        np.isfinite(points_3d[:, :, 2]) &
        (points_3d[:, :, 2] > min_depth_mm) &
        (points_3d[:, :, 2] < max_depth_mm)
    )

    xyz = points_3d[mask].astype(np.float32)  # (N, 3) - tylko wazne punkty

    # Diagnostyka odrzuconych punktow: rozdzielamy powody, by latwiej zlokalizowac
    # problem (brak dysparycji vs. obciecie zakresem glebokosci).
    z = points_3d[:, :, 2]
    has_disp = disparity > 0
    finite = has_disp & np.isfinite(z)
    n_disp = int(has_disp.sum())
    n_near = int((finite & (z <= min_depth_mm)).sum())
    n_far = int((finite & (z >= max_depth_mm)).sum())

    colors = None
    if color_image is not None:
        # OpenCV przechowuje obrazy w formacie BGR; konwertujemy do RGB
        # bo format PLY i wiekszosc narzedzi 3D oczekuje RGB
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) if color_image.ndim == 3 else \
              cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
        colors = rgb[mask]  # (N, 3) uint8
    elif color_image is None:
        log.debug("build_pointcloud: brak color_image - chmura bez kolorow")

    log.info("Chmura punktow: %d punktow (maska %.1f%% pikseli, zakres %.0f-%.0f mm)",
             len(xyz), 100 * mask.sum() / mask.size, min_depth_mm, max_depth_mm)
    log.debug("Odrzucono: razem=%d, za blisko <=%.0f mm=%d, za daleko >=%.0f mm=%d",
              n_disp - len(xyz), min_depth_mm, n_near, max_depth_mm, n_far)
    if len(xyz) == 0:
        log.warning("Chmura punktow PUSTA - brak punktow 3D w zakresie %.0f-%.0f mm "
                    "(dysparycja>0: %d). Sprawdz dysparycje, macierz Q i progi glebokosci.",
                    min_depth_mm, max_depth_mm, n_disp)
    elif len(xyz) < 100:
        log.warning("Chmura punktow bardzo rzadka: %d punktow - detekcja palety moze sie nie udac",
                    len(xyz))
    return xyz, colors


def filter_pointcloud(
    xyz: np.ndarray,
    colors: np.ndarray | None = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filtr statystyczny: usuwa punkty oddalone o std_ratio * std od srednich odleglosci.

    Prosta implementacja bez Open3D - dziala na samym NumPy.
    Usuwa typowe "latajace" punkty na krawedziach obiektow.

    Args:
        xyz:          (N, 3) chmura punktow
        colors:       (N, 3) kolory lub None
        nb_neighbors: liczba najblizszych sasiadow do liczenia srednich odleglosci
        std_ratio:    prog: punkty z odlegloscia > mean + std_ratio*std sa usuwane

    Returns:
        xyz_filt, colors_filt - przefiltrowana chmura
    """
    if len(xyz) < nb_neighbors + 1:
        log.warning("Za malo punktow do filtrowania (%d), pomijam", len(xyz))
        return xyz, colors

    # Przyblizony filtr statystyczny: szacujemy srednia odleglosc do k najblizszych
    # sasiadow na probce, a nastepnie usuwamy punkty, ktorych odleglosc od probki
    # przekracza prog mean + std_ratio*std. Eliminuje "latajace" punkty szumu SGBM
    # bez potrzeby biblioteki Open3D.
    sample_size = min(len(xyz), 5000)
    sample_idx  = np.random.choice(len(xyz), sample_size, replace=False)
    sample      = xyz[sample_idx]

    # Obliczamy srednia odleglosc do nb_neighbors najblizszych sasiadow
    # dla kazdego punktu w probce. Przetwarzamy partiami (CHUNK), zeby
    # uniknac alokacji macierzy (sample_size x sample_size) w pamieci.
    CHUNK = 500
    knn_dist = np.zeros(sample_size)
    for i in range(0, sample_size, CHUNK):
        chunk = sample[i:i+CHUNK]                              # (C, 3)
        diff  = chunk[:, None, :] - sample[None, :, :]        # (C, S, 3)
        dist  = np.sqrt((diff**2).sum(axis=2))                 # (C, S)
        dist[:, i:i+len(chunk)] = np.inf                       # wyklucz siebie (dystans=0)
        knn_dist[i:i+CHUNK] = np.sort(dist, axis=1)[:, :nb_neighbors].mean(axis=1)

    mean_d = knn_dist.mean()
    std_d  = knn_dist.std()
    # Prog: punkty dalej niz mean + std_ratio*std od swoich sasiadow sa uznawane za szum
    thresh = mean_d + std_ratio * std_d

    # Dla pelnej chmury liczymy odleglosc do najblizszego punktu w probce
    # (partiami) i porownujemy z progiem
    min_dist = np.full(len(xyz), np.inf)
    for i in range(0, len(xyz), CHUNK):
        chunk = xyz[i:i+CHUNK]                                 # (C, 3)
        diff  = chunk[:, None, :] - sample[None, :, :]        # (C, S, 3)
        dist  = np.sqrt((diff**2).sum(axis=2))                 # (C, S)
        min_dist[i:i+CHUNK] = dist.min(axis=1)

    keep = min_dist < thresh
    kept_pct = 100 * keep.mean()
    log.info("Filtr statystyczny: zachowano %d/%d punktow (%.0f%%, prog=%.1f mm)",
             keep.sum(), len(xyz), kept_pct, thresh)
    # Usuniecie wiekszosci punktow oznacza zwykle silnie poszarpana chmure
    # (slaba tekstura / zla rektyfikacja) - dane wyjsciowe moga byc niewiarygodne.
    if kept_pct < 50.0:
        log.warning("Filtr statystyczny usunal %.0f%% punktow - chmura bardzo zaszumiona, "
                    "wyniki pomiaru moga byc niepewne", 100 - kept_pct)

    return xyz[keep], (colors[keep] if colors is not None else None)


# ---------------------------------------------------------------------------
# Zapis PLY (dziala bez Open3D - czysty Python/NumPy)
# ---------------------------------------------------------------------------

def _ply_header(n: int, has_color: bool, binary: bool) -> str:
    """Buduje naglowek pliku PLY wspolny dla wersji ASCII i binarnej."""
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = f"ply\nformat {fmt}\nelement vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    return header + "end_header\n"


def save_ply(path: str, xyz: np.ndarray, colors: np.ndarray | None = None):
    """Zapisuje chmure punktow do pliku PLY (ASCII).

    Format PLY jest obslugiwany przez MeshLab, CloudCompare, Open3D,
    Blender i wiekszos narzedzi do wizualizacji 3D.

    Args:
        path:   sciezka wyjsciowa (np. "cloud.ply")
        xyz:    (N, 3) float32 wspolrzedne w mm
        colors: (N, 3) uint8 RGB lub None
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    has_color = colors is not None

    with open(path, "w") as f:
        f.write(_ply_header(n, has_color, binary=False))
        for i in range(n):
            x, y, z = xyz[i]
            line = f"{x:.3f} {y:.3f} {z:.3f}"
            if has_color:
                r, g, b = colors[i]
                line += f" {int(r)} {int(g)} {int(b)}"
            f.write(line + "\n")

    log.info("Zapisano PLY: %s (%d punktow)", path, n)


def save_ply_binary(path: str, xyz: np.ndarray, colors: np.ndarray | None = None):
    """Zapisuje chmure punktow do pliku PLY w formacie binarnym little-endian.

    Znacznie szybszy i produkuje mniejsze pliki niz wersja ASCII.
    Preferowany dla duzych chmur (>100k punktow). Kompatybilny z MeshLab,
    CloudCompare i Open3D.

    Args:
        path:   sciezka wyjsciowa
        xyz:    (N, 3) float32 wspolrzedne w mm
        colors: (N, 3) uint8 RGB lub None
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    has_color = colors is not None

    with open(path, "wb") as f:
        f.write(_ply_header(n, has_color, binary=True).encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *xyz[i].tolist()))
            if has_color:
                f.write(struct.pack("BBB", *colors[i].tolist()))

    log.info("Zapisano PLY (binary): %s (%d punktow)", path, n)


# ---------------------------------------------------------------------------
# Prosta wizualizacja 2D chmury (rzut z gory i z boku) - bez Open3D
# ---------------------------------------------------------------------------

def _render_projection(
    xyz: np.ndarray,
    colors: np.ndarray | None,
    h_axis: int,
    v_axis: int,
    canvas_size: int = 600,
) -> np.ndarray:
    """Rzutuje chmure na wybrana pare osi i zwraca obraz BGR.

    Os pozioma obrazu = h_axis, os pionowa = v_axis (odwrocona, by wieksze
    wartosci byly u gory). Wspolrzedne sa normalizowane do [0, canvas_size-1].
    Operacja jest w pelni wektorowa - przy kolizji pikseli wygrywa ostatni punkt.

    Args:
        xyz:         (N, 3) chmura punktow [mm]
        colors:      (N, 3) kolory RGB lub None (szary)
        h_axis:      indeks osi (0=X,1=Y,2=Z) odwzorowanej na poziom obrazu
        v_axis:      indeks osi odwzorowanej na pion obrazu
        canvas_size: rozmiar obrazu wyjsciowego [px]
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if len(xyz) == 0:
        return canvas

    h, v = xyz[:, h_axis], xyz[:, v_axis]
    h_norm = ((h - h.min()) / max(h.max() - h.min(), 1e-3) * (canvas_size - 1)).astype(int)
    v_norm = ((v - v.min()) / max(v.max() - v.min(), 1e-3) * (canvas_size - 1)).astype(int)
    v_norm = canvas_size - 1 - v_norm  # wieksze wartosci osi -> u gory obrazu

    if colors is not None:
        canvas[v_norm, h_norm] = colors[:, ::-1]  # RGB -> BGR
    else:
        canvas[v_norm, h_norm] = (200, 200, 200)
    return canvas


def render_topdown(
    xyz: np.ndarray,
    colors: np.ndarray | None = None,
    canvas_size: int = 600,
) -> np.ndarray:
    """Rzut z gory (plaszczyzna X-Z): X -> poziom, Z (glebokos) -> pion."""
    return _render_projection(xyz, colors, h_axis=0, v_axis=2, canvas_size=canvas_size)


def render_sideview(
    xyz: np.ndarray,
    colors: np.ndarray | None = None,
    canvas_size: int = 600,
) -> np.ndarray:
    """Rzut z boku (plaszczyzna X-Y): X -> poziom, Y (wysokos) -> pion."""
    return _render_projection(xyz, colors, h_axis=0, v_axis=1, canvas_size=canvas_size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from logging_setup import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Generowanie chmury punktow 3D")
    parser.add_argument("--calib",      default="calib_output/stereo.json")
    parser.add_argument("--disparity",  required=True,
                        help="Plik .npy z mapa dysparycji (z disparity.py)")
    parser.add_argument("--color",      default=None,
                        help="Obraz BGR do kolorowania punktow (np. left_rect.png)")
    parser.add_argument("--output",     default="depth_output/cloud.ply")
    parser.add_argument("--max-depth",  type=float, default=5000.0)
    parser.add_argument("--min-depth",  type=float, default=50.0)
    parser.add_argument("--no-filter",  action="store_true",
                        help="Pomijn filtr statystyczny")
    parser.add_argument("--binary",     action="store_true",
                        help="Zapisz PLY w formacie binarnym")
    args = parser.parse_args()

    stereo = load_params(args.calib, stereo=True)
    disp   = np.load(args.disparity)
    color_img = cv2.imread(args.color) if args.color else None

    xyz, colors = build_pointcloud(
        disp, stereo.Q, color_img, args.max_depth, args.min_depth
    )

    if not args.no_filter:
        xyz, colors = filter_pointcloud(xyz, colors)

    out_path = args.output
    if args.binary:
        save_ply_binary(out_path, xyz, colors)
    else:
        save_ply(out_path, xyz, colors)

    # Zapisz rzuty 2D
    out_dir = Path(out_path).parent
    top  = render_topdown(xyz, colors)
    side = render_sideview(xyz, colors)
    cv2.imwrite(str(out_dir / "view_topdown.png"), top)
    cv2.imwrite(str(out_dir / "view_sideview.png"), side)
    log.info("Rzuty 2D zapisane do: %s", out_dir)