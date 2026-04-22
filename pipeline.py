# pipeline.py
"""Pelny pipeline stereo: kalibracja -> rektyfikacja -> dysparycja -> glebokos -> chmura.

Dziala w dwoch trybach:
  1. SYNTETYCZNY (domyslny) - generuje wirtualna scene z pudlkami, nie wymaga kamer.
     Sluzy do weryfikacji kodu przed sesja zdjeciowa.
  2. PRAWDZIWY   (--mode real) - wczytuje prawdziwa pare stereo i kalibracje z dysku.

Uzycie:
    # Tryb syntetyczny - dziala od razu, bez zdiec
    python pipeline.py

    # Tryb rzeczywisty - po zrobieniu zdiec i kalibracji
    python pipeline.py --mode real \
        --calib calib_output/stereo.json \
        --left  stereo_pairs/left_000.png \
        --right stereo_pairs/right_000.png
"""
import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from calibration import (
    StereoParams, CameraParams,
    calibrate_stereo, save_params, load_params,
    get_image_paths,
)
from disparity import (
    rectify_pair, compute_disparity, disparity_to_depth,
    colormap_disparity, colormap_depth, draw_epipolar_check,
    SGBMConfig,
)
from pointcloud import (
    build_pointcloud, filter_pointcloud,
    save_ply, render_topdown, render_sideview,
)
from pallet import detect_pallet
from measurement import measure_object, validate_measurement, generate_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# DANE SYNTETYCZNE
# ===========================================================================

def _make_synthetic_stereo_params(
    img_w: int = 640,
    img_h: int = 480,
    baseline_mm: float = 120.0,
    focal_mm: float = 800.0,
) -> StereoParams:
    """Tworzy StereoParams z typowymi parametrami telefonow (bez kalibracji).

    Symuluje dwa telefony ustawione poziomo ~12 cm od siebie.
    Macierz Q jest obliczana analitycznie z baseline i ogniskowej.
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[focal_mm, 0, cx], [0, focal_mm, cy], [0, 0, 1.0]])
    dist = np.zeros(5)

    left_cam  = CameraParams(K.copy(), dist.copy(), 0.5, (img_w, img_h))
    right_cam = CameraParams(K.copy(), dist.copy(), 0.5, (img_w, img_h))

    R = np.eye(3)
    T = np.array([[baseline_mm], [0.0], [0.0]])

    # Przy identycznych kamerach R1=R2=I, P1=P2=K|0, Q jest analityczne
    R1 = R2 = np.eye(3)
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K, np.array([[-baseline_mm * focal_mm], [0], [0]])])
    # Q: standardowa macierz reprojekcji dla konfiguracji poziomej
    # Znak -1/baseline daje Z > 0 gdy dysparycja > 0 i kamera prawa jest
    # przesunieta w prawo (T[0] > 0, co odpowiada konwencji OpenCV).
    Q = np.array([
        [1, 0,           0,       -cx],
        [0, 1,           0,       -cy],
        [0, 0,           0,   focal_mm],
        [0, 0, 1/baseline_mm,       0],
    ], dtype=np.float64)

    return StereoParams(
        left=left_cam, right=right_cam,
        R=R, T=T,
        E=np.eye(3), F=np.eye(3),
        reproj_error=0.0,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    )


def _render_box(
    depth_map: np.ndarray,
    img: np.ndarray,
    x0: int, y0: int,
    w: int, h: int,
    depth_mm: float,
    color_bgr: tuple,
):
    """Rysuje prostokat na obrazie i przypisuje mu glebokos na mapie."""
    img[y0:y0+h, x0:x0+w] = color_bgr
    depth_map[y0:y0+h, x0:x0+w] = depth_mm


def generate_synthetic_scene(
    img_w: int = 640,
    img_h: int = 480,
    baseline_mm: float = 120.0,
    focal_mm: float = 800.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StereoParams]:
    """Generuje syntetyczna scene: dwa obrazy stereo + mape glebokosci GT + parametry.

    Scena zawiera 3 prostokatne "pudla" na jednolitym tle, kazde na innej
    glebokosci. Dysparycja jest obliczana analitycznie: d = f * B / Z.

    Returns:
        left_img, right_img, depth_gt_mm, stereo_params
    """
    stereo = _make_synthetic_stereo_params(img_w, img_h, baseline_mm, focal_mm)
    cx = img_w / 2.0

    # Definicje pudel: (x0, y0, szerokosc, wysokosc, glebokos_mm, kolor_bgr)
    boxes = [
        (80,  100, 180, 200, 800.0,  (60,  100, 200)),   # lewy - blizej, niebieski
        (260, 150, 160, 160, 1200.0, (60,  180,  60)),   # srodkowy, zielony
        (440, 80,  130, 220, 1600.0, (200,  60,  60)),   # prawy - dalej, czerwony
    ]

    # Tlo (szare, "dalej" niz wszystkie pudla)
    bg_depth = 3000.0
    left_img  = np.full((img_h, img_w, 3), 40, dtype=np.uint8)
    right_img = np.full((img_h, img_w, 3), 40, dtype=np.uint8)
    depth_gt  = np.full((img_h, img_w), bg_depth, dtype=np.float32)

    # Dysparycja tla: d = f * B / Z
    bg_disp = focal_mm * baseline_mm / bg_depth

    for x0, y0, w, h, z_mm, color in boxes:
        disp_mm = focal_mm * baseline_mm / z_mm
        shift_px = int(round(disp_mm))  # przesuniecie w pikselach dla prawego obrazu

        _render_box(depth_gt,  left_img,  x0,            y0, w, h, z_mm,   color)
        _render_box(depth_gt,  right_img, x0 - shift_px, y0, w, h, z_mm,   color)

    # Dodaj lekki szum Gaussa (symulacja szumu sensora)
    noise_l = np.random.RandomState(0).randint(-8, 8, left_img.shape).astype(np.int16)
    noise_r = np.random.RandomState(1).randint(-8, 8, right_img.shape).astype(np.int16)
    left_img  = np.clip(left_img.astype(np.int16)  + noise_l, 0, 255).astype(np.uint8)
    right_img = np.clip(right_img.astype(np.int16) + noise_r, 0, 255).astype(np.uint8)

    log.info("Scena syntetyczna: %dx%d px, %d pudel, baza=%.0f mm",
             img_w, img_h, len(boxes), baseline_mm)
    return left_img, right_img, depth_gt, stereo


# ===========================================================================
# WNIOSKI - automatyczne podsumowanie wynikow
# ===========================================================================

def compute_depth_error(
    depth_est: np.ndarray,
    depth_gt: np.ndarray,
    max_depth_mm: float = 4000.0,
) -> dict:
    """Liczy blad oszacowania glebokosci wzgledem ground-truth (tylko tryb syntetyczny).

    Zwraca slownik z metrykami do raportu.
    """
    valid = (depth_gt > 0) & (depth_gt < max_depth_mm) & (depth_est > 0)
    if not valid.any():
        return {"uwaga": "brak wspolnych pikseli do porownania"}

    err = np.abs(depth_est[valid] - depth_gt[valid])
    rel = err / depth_gt[valid]

    return {
        "n_pikseli":         int(valid.sum()),
        "mae_mm":            float(err.mean()),
        "medae_mm":          float(np.median(err)),
        "rmse_mm":           float(np.sqrt((err**2).mean())),
        "blad_wzgledny_pct": float(100 * rel.mean()),
        "pokrycie_pct":      float(100 * valid.mean()),
    }


def print_report(metrics: dict, stereo: StereoParams, mode: str):
    """Drukuje wnioski w formacie gotowym do przeklejenia do raportu."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RAPORT M2 - WYNIKI PIPELINE STEREO")
    print(sep)
    print(f"\nTryb:         {mode}")
    print(f"Blad reproj.: {stereo.reproj_error:.4f} px")
    print(f"Baza stereo:  |T| = {float(np.linalg.norm(stereo.T)):.1f} mm")
    print(f"Ogniskowa L:  fx={stereo.left.camera_matrix[0,0]:.1f}  fy={stereo.left.camera_matrix[1,1]:.1f}")
    print(f"Ogniskowa P:  fx={stereo.right.camera_matrix[0,0]:.1f}  fy={stereo.right.camera_matrix[1,1]:.1f}")

    if metrics:
        print("\n--- Dokladnosc mapy glebokosci ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<25s}: {v:.2f}")
            else:
                print(f"  {k:<25s}: {v}")

    print(f"\n{sep}")
    print("WNIOSKI:")
    print("""
  1. Kalibracja stereo metoda Zhanga (OpenCV) pozwolila wyznaczyc
     parametry wewnetrzne obu kamer oraz wzajemne polozenie (R, T).

  2. Rektyfikacja (cv2.remap z macierzami R1/R2/P1/P2) wyrownuje
     linie epipolarne - odpowiadajace sobie punkty leza w tych samych
     wierszach obu obrazow, co jest warunkiem koniecznym dla SGBM.

  3. Semi-Global Block Matching (SGBM) oblicza mape dysparycji przez
     porownanie blokow pikseli i minimalizacje kosztu globalnego.
     Jakos mapy zalezy od tekstury obiektu - gladkie, jednolite
     powierzchnie daja obszary bez dysparycji (dziury w mapie).

  4. Macierz Q z cv2.stereoRectify pozwala przeliczys dysparycje
     na glebokos (mm) przez cv2.reprojectImageTo3D - jedna operacja
     macierzowa dla calego obrazu.

  5. Glowne zrodla bledu:
     - szum sensora telefonow (brak globalnego migawki)
     - brak sprzętowej synchronizacji klatek
     - ograniczona tekstura obiektow (kartony)
     - efekty krawedziowe SGBM (halo wokol obiektow)
""")
    print(sep)


# ===========================================================================
# GLOWNY PIPELINE
# ===========================================================================

def run_pipeline(
    mode: str,
    calib_path: str | None,
    left_path: str | None,
    right_path: str | None,
    output_dir: str,
    max_depth_mm: float,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Wczytanie / wygenerowanie danych
    # ------------------------------------------------------------------
    if mode == "synthetic":
        log.info("=== TRYB SYNTETYCZNY ===")
        left_img, right_img, depth_gt, stereo = generate_synthetic_scene()
        metrics_possible = True
    else:
        log.info("=== TRYB RZECZYWISTY ===")
        if not calib_path or not left_path or not right_path:
            raise SystemExit("Tryb real wymaga --calib, --left, --right")
        stereo = load_params(calib_path, stereo=True)
        left_img  = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            raise SystemExit(f"Nie mozna wczytac obrazow: {left_path}, {right_path}")
        depth_gt = None
        metrics_possible = False
        log.info("Wczytano: %s | %s", left_path, right_path)

    # Zapis obrazow wejsciowych
    cv2.imwrite(str(out / "input_left.png"),  left_img)
    cv2.imwrite(str(out / "input_right.png"), right_img)

    # ------------------------------------------------------------------
    # 2. Rektyfikacja (stereo.rectify_maps() z calibration.py Kuby)
    # ------------------------------------------------------------------
    log.info("--- Rektyfikacja ---")
    img_size = (left_img.shape[1], left_img.shape[0])
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img, img_size)

    cv2.imwrite(str(out / "left_rect.png"),  left_rect)
    cv2.imwrite(str(out / "right_rect.png"), right_rect)
    cv2.imwrite(str(out / "epipolar_check.png"),
                draw_epipolar_check(left_rect, right_rect))

    # ------------------------------------------------------------------
    # 3. Mapa dysparycji - SGBM
    # ------------------------------------------------------------------
    log.info("--- Mapa dysparycji (SGBM) ---")
    cfg = SGBMConfig(
        num_disparities=64,
        block_size=7,
        uniqueness=10,
        speckle_window=100,
    )
    disp = compute_disparity(left_rect, right_rect, cfg)

    np.save(str(out / "disparity.npy"), disp)
    cv2.imwrite(str(out / "disparity_color.png"), colormap_disparity(disp))

    # ------------------------------------------------------------------
    # 4. Mapa glebokosci (stereo.Q z StereoParams Kuby)
    # ------------------------------------------------------------------
    log.info("--- Mapa glebokosci ---")
    depth = disparity_to_depth(disp, stereo.Q, max_depth_mm)

    np.save(str(out / "depth_mm.npy"), depth)
    cv2.imwrite(str(out / "depth_color.png"), colormap_depth(depth))

    # ------------------------------------------------------------------
    # 5. Chmura punktow
    # ------------------------------------------------------------------
    log.info("--- Chmura punktow ---")
    xyz, colors = build_pointcloud(disp, stereo.Q, left_rect, max_depth_mm)
    xyz, colors = filter_pointcloud(xyz, colors)

    save_ply(str(out / "cloud.ply"), xyz, colors)
    cv2.imwrite(str(out / "view_topdown.png"),  render_topdown(xyz, colors))
    cv2.imwrite(str(out / "view_sideview.png"), render_sideview(xyz, colors))

    # ------------------------------------------------------------------
    # ETAP 5-7: Detekcja palety, segmentacja, pomiar wymiarow
    # ------------------------------------------------------------------
    try:
        log.info("--- ETAP 5: Detekcja europalety (RANSAC) ---")
        pallet_result = detect_pallet(
            xyz,
            n_iterations=1000,
            distance_threshold=10.0,
            min_inliers=50,
        )
        log.info(
            "Paleta: RMS=%.1f mm, %d inlierow, %d pkt w ROI",
            pallet_result.plane.rms_residual,
            pallet_result.plane.inlier_mask.sum(),
            pallet_result.roi_mask.sum(),
        )

        log.info("--- ETAP 6+7: Segmentacja i pomiar obiektu ---")
        meas = measure_object(xyz, pallet_result, noise_floor_mm=20.0)
        validation = validate_measurement(meas)
        report_text = generate_report(meas, validation)
        print(report_text)

        with open(str(out / "measurement_report.txt"), "w") as f:
            f.write(report_text)
        log.info("Raport zapisany: %s", out / "measurement_report.txt")

    except (RuntimeError, ValueError) as e:
        log.warning("Detekcja palety / pomiar nieudane: %s", e)

    # ------------------------------------------------------------------
    # 6. Metryki i wnioski
    # ------------------------------------------------------------------
    metrics = {}
    if metrics_possible and depth_gt is not None:
        metrics = compute_depth_error(depth, depth_gt, max_depth_mm)

    print_report(metrics, stereo, mode)

    # Zapisz metryki do pliku
    with open(str(out / "report.txt"), "w") as f:
        f.write(f"Tryb: {mode}\n")
        f.write(f"Blad reproj.: {stereo.reproj_error:.4f} px\n")
        f.write(f"|T| = {float(np.linalg.norm(stereo.T)):.1f} mm\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    log.info("=== GOTOWE - wyniki w: %s ===", out)
    print(f"\nPliki wynikowe:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline stereo: rektyfikacja -> dysparycja -> glebokos -> chmura"
    )
    parser.add_argument("--mode",      choices=["synthetic", "real"], default="synthetic",
                        help="synthetic = dane testowe, real = prawdziwe zdjecia")
    parser.add_argument("--calib",     default="calib_output/stereo.json",
                        help="JSON z kalibracja stereo (tryb real)")
    parser.add_argument("--left",      help="Obraz lewej kamery (tryb real)")
    parser.add_argument("--right",     help="Obraz prawej kamery (tryb real)")
    parser.add_argument("--output",    default="./pipeline_output",
                        help="Katalog na wszystkie wyniki")
    parser.add_argument("--max-depth", type=float, default=5000.0,
                        help="Maks. glebokos [mm]")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        calib_path=args.calib,
        left_path=args.left,
        right_path=args.right,
        output_dir=args.output,
        max_depth_mm=args.max_depth,
    )
