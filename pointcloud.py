# pointcloud.py
"""Generowanie i zapis chmury punktow 3D z mapy dysparycji.

Uzywa macierzy Q z StereoParams Kuby (stereo.Q) do reprojekcji
dysparycji w przestrzen 3D, opcjonalnie koloruje punkty obrazem RGB.

Uzycie:
    python pointcloud.py --calib calib_output/stereo.json \
                         --disparity depth_output/disparity.npy \
                         --color left_rect.png \
                         --output depth_output/cloud.ply

Z kodu:
    from pointcloud import build_pointcloud, save_ply, filter_pointcloud
    xyz, colors = build_pointcloud(disparity, stereo.Q, left_rect)
    save_ply("cloud.ply", xyz, colors)
"""
import logging
import struct
import numpy as np
import cv2
from pathlib import Path

from calibration import StereoParams, load_params

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budowanie chmury punktow
# ---------------------------------------------------------------------------

def build_pointcloud(
    disparity: np.ndarray,
    Q: np.ndarray,
    color_image: np.ndarray | None = None,
    max_depth_mm: float = 5000.0,
    min_depth_mm: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Buduje chmure punktow 3D z mapy dysparycji i macierzy Q.

    Macierz Q pochodzi z StereoParams.Q (zapisana przez Kube po stereoRectify).
    cv2.reprojectImageTo3D mnozy kazdy piksel [x, y, d, 1]^T przez Q i zwraca
    wspolrzedne [X, Y, Z] w mm (jesli kalibracja byla w mm).

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
    points_3d = cv2.reprojectImageTo3D(disparity, Q)  # (H, W, 3)

    # Maska: tylko piksele z prawidlowa dysparycja i sensowna glebokoscia
    mask = (
        (disparity > 0) &
        np.isfinite(points_3d[:, :, 2]) &
        (points_3d[:, :, 2] > min_depth_mm) &
        (points_3d[:, :, 2] < max_depth_mm)
    )

    xyz = points_3d[mask].astype(np.float32)  # (N, 3)

    colors = None
    if color_image is not None:
        # color_image jest BGR (OpenCV), zapisujemy jako RGB
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) if color_image.ndim == 3 else \
              cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
        colors = rgb[mask]  # (N, 3) uint8

    log.info("Chmura punktow: %d punktow (maska %.1f%% pikseli)",
             len(xyz), 100 * mask.sum() / mask.size)
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

    # Przyblizony filtr: dla kazdego punktu liczymy odleglosc do nb_neighbors
    # najblizszych sasiadow. Uzywamy losowej probki dla wydajnosci.
    sample_size = min(len(xyz), 5000)
    sample_idx  = np.random.choice(len(xyz), sample_size, replace=False)
    sample      = xyz[sample_idx]

    # Odleglosci w probce - partiami zeby uniknac macierzy S*S w pamieci
    CHUNK = 500
    knn_dist = np.zeros(sample_size)
    for i in range(0, sample_size, CHUNK):
        chunk = sample[i:i+CHUNK]                              # (C, 3)
        diff  = chunk[:, None, :] - sample[None, :, :]        # (C, S, 3)
        dist  = np.sqrt((diff**2).sum(axis=2))                 # (C, S)
        dist[:, i:i+len(chunk)] = np.inf                       # wyklucz siebie
        knn_dist[i:i+CHUNK] = np.sort(dist, axis=1)[:, :nb_neighbors].mean(axis=1)

    mean_d = knn_dist.mean()
    std_d  = knn_dist.std()
    thresh = mean_d + std_ratio * std_d

    # Prog dla pelnej chmury: odleglosc do najblizszego punkta w probce (partiami)
    min_dist = np.full(len(xyz), np.inf)
    for i in range(0, len(xyz), CHUNK):
        chunk = xyz[i:i+CHUNK]                                 # (C, 3)
        diff  = chunk[:, None, :] - sample[None, :, :]        # (C, S, 3)
        dist  = np.sqrt((diff**2).sum(axis=2))                 # (C, S)
        min_dist[i:i+CHUNK] = dist.min(axis=1)

    keep = min_dist < thresh
    log.info("Filtr statystyczny: zachowano %d/%d punktow (%.0f%%)",
             keep.sum(), len(xyz), 100 * keep.mean())

    return xyz[keep], (colors[keep] if colors is not None else None)


# ---------------------------------------------------------------------------
# Zapis PLY (dziala bez Open3D - czysty Python/NumPy)
# ---------------------------------------------------------------------------

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
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            x, y, z = xyz[i]
            line = f"{x:.3f} {y:.3f} {z:.3f}"
            if has_color:
                r, g, b = colors[i]
                line += f" {int(r)} {int(g)} {int(b)}"
            f.write(line + "\n")

    log.info("Zapisano PLY: %s (%d punktow)", path, n)


def save_ply_binary(path: str, xyz: np.ndarray, colors: np.ndarray | None = None):
    """Zapisuje chmure punktow do pliku PLY w formacie binarnym (szybsze, mniejsze pliki).

    Preferowany dla duzych chmur (>100k punktow).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    has_color = colors is not None

    with open(path, "wb") as f:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        if has_color:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))

        for i in range(n):
            f.write(struct.pack("<fff", *xyz[i].tolist()))
            if has_color:
                f.write(struct.pack("BBB", *colors[i].tolist()))

    log.info("Zapisano PLY (binary): %s (%d punktow)", path, n)


# ---------------------------------------------------------------------------
# Prosta wizualizacja 2D chmury (rzut z gory i z boku) - bez Open3D
# ---------------------------------------------------------------------------

def render_topdown(
    xyz: np.ndarray,
    colors: np.ndarray | None = None,
    resolution_mm: float = 5.0,
    canvas_size: int = 600,
) -> np.ndarray:
    """Rzut chmury punktow z gory (os X-Z) jako obraz BGR.

    Uzyteczne do szybkiej weryfikacji ksztaltu obiektu bez narzedzi 3D.
    """
    if len(xyz) == 0:
        return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    x, z = xyz[:, 0], xyz[:, 2]
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Normalizacja do pikseli
    x_norm = ((x - x.min()) / max(x.max()-x.min(), 1e-3) * (canvas_size - 1)).astype(int)
    z_norm = ((z - z.min()) / max(z.max()-z.min(), 1e-3) * (canvas_size - 1)).astype(int)
    z_norm = canvas_size - 1 - z_norm  # odwrocenie osi Y

    for i in range(len(xyz)):
        px, py = x_norm[i], z_norm[i]
        if 0 <= px < canvas_size and 0 <= py < canvas_size:
            if colors is not None:
                r, g, b = colors[i]
                canvas[py, px] = [int(b), int(g), int(r)]  # BGR
            else:
                canvas[py, px] = [200, 200, 200]

    return canvas


def render_sideview(
    xyz: np.ndarray,
    colors: np.ndarray | None = None,
    canvas_size: int = 600,
) -> np.ndarray:
    """Rzut chmury z boku (os X-Y, gdzie Y to wysokosc) jako obraz BGR."""
    if len(xyz) == 0:
        return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    x, y = xyz[:, 0], xyz[:, 1]
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    x_norm = ((x - x.min()) / max(x.max()-x.min(), 1e-3) * (canvas_size - 1)).astype(int)
    y_norm = ((y - y.min()) / max(y.max()-y.min(), 1e-3) * (canvas_size - 1)).astype(int)
    y_norm = canvas_size - 1 - y_norm

    for i in range(len(xyz)):
        px, py = x_norm[i], y_norm[i]
        if 0 <= px < canvas_size and 0 <= py < canvas_size:
            if colors is not None:
                r, g, b = colors[i]
                canvas[py, px] = [int(b), int(g), int(r)]
            else:
                canvas[py, px] = [200, 200, 200]

    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
