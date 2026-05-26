# pipeline.py
"""Pelny pipeline stereo - laczy wszystkie etapy od par zdjecien do pomiaru wymiarow.

Tryby pracy:
  1. SYNTETYCZNY (domyslny) - generuje wirtualna scene z pudlkami, nie wymaga kamer.
     Sluzy do weryfikacji poprawnosci kodu i kalibracji przed sesja zdjeciowa.
  2. RZECZYWISTY (--mode real) - wczytuje prawdziwa pare stereo i kalibracje z dysku.

Kolejnosc etapow:
  1. Wczytanie / wygenerowanie pary stereo
  2. Rektyfikacja - wyrownanie linii epipolarnych
  3. Mapa dysparycji - SGBM
  4. Mapa glebokosci - reprojekcja przez macierz Q
  5. Chmura punktow - build + filtrowanie
  6. Detekcja palety - RANSAC + SVD
  7. Pomiar obiektu - segmentacja + bounding box
  8. Raport i metryki

Uzycie:
    # Tryb syntetyczny - dziala od razu, bez zdiec
    python pipeline.py

    # Tryb rzeczywisty - po wykonaniu kalibracji i zdjeciach
    python pipeline.py --mode real \\
        --calib calib_output/stereo.json \\
        --left  stereo_pairs/left_000.png \\
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
from logging_setup import setup_logging
import config

setup_logging()
log = logging.getLogger(__name__)


def _save_pallet_debug(path, pallet_result, meas, noise_floor_mm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        log.warning("matplotlib niedostepny — pominięto pallet_debug.png")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_p = pallet_result.xyz_pallet
    roi_mask = pallet_result.roi_mask
    inlier_mask = pallet_result.plane.inlier_mask
    xyz_roi = xyz_p[roi_mask]
    xyz_inliers = xyz_p[inlier_mask]
    xyz_obj = meas.object_pts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if len(xyz_roi):
        ax1.scatter(xyz_roi[:, 0], xyz_roi[:, 1], s=0.3, c="lightgray", label="ROI")
    ax1.scatter(xyz_inliers[:, 0], xyz_inliers[:, 1], s=0.5, c="green",
                alpha=0.5, label="Paleta (inlierzy)")
    if len(xyz_obj):
        ax1.scatter(xyz_obj[:, 0], xyz_obj[:, 1], s=2, c="red", label="Obiekt")
    w, l = config.PALLET_WIDTH_MM, config.PALLET_LENGTH_MM
    ax1.add_patch(patches.Rectangle((-w / 2, -l / 2), w, l, linewidth=1.5,
                  edgecolor="blue", facecolor="none", linestyle="--", label="ROI config"))
    ax1.set_aspect("equal")
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_title(
        f"Widok z góry — RMS={pallet_result.plane.rms_residual:.1f} mm, "
        f"{inlier_mask.sum()} inlierów"
    )
    ax1.legend(markerscale=5, fontsize=8)

    if len(xyz_roi):
        ax2.scatter(xyz_roi[:, 0], xyz_roi[:, 2], s=0.3, c="lightgray", label="ROI")
    if len(xyz_obj):
        ax2.scatter(xyz_obj[:, 0], xyz_obj[:, 2], s=2, c="red", label="Obiekt")
    ax2.axhline(0, color="green", linewidth=1.2, label="Powierzchnia (Z=0)")
    ax2.axhline(noise_floor_mm, color="orange", linewidth=1, linestyle="--",
                label=f"noise_floor={noise_floor_mm:.0f} mm")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Z — wysokość [mm]")
    ax2.set_title("Widok z boku (XZ)")
    ax2.legend(markerscale=5, fontsize=8)

    plt.tight_layout()
    plt.savefig(str(path), dpi=100)
    plt.close(fig)
    log.info("Pallet debug zapisany: %s", path)


# ===========================================================================
# DANE SYNTETYCZNE
# ===========================================================================

def _make_synthetic_stereo_params(
    img_w: int = 640,
    img_h: int = 480,
    baseline_mm: float = 120.0,
    focal_mm: float = 800.0,
) -> StereoParams:
    """Tworzy StereoParams z typowymi parametrami telefonow bez kalibracji.

    Symuluje dwa identyczne telefony ustawione poziomo ~12 cm od siebie
    (standardowa baza dla systemu dwukamerowego). Parametry sa typowe
    dla telefonow z aparatem o ogniskowej ~28mm ekwiwalentu pelnej klatki.
    Macierz Q jest obliczana analitycznie - nie wymaga cv2.stereoRectify.
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[focal_mm, 0, cx], [0, focal_mm, cy], [0, 0, 1.0]])
    dist = np.zeros(5)

    left_cam  = CameraParams(K.copy(), dist.copy(), 0.5, (img_w, img_h))
    right_cam = CameraParams(K.copy(), dist.copy(), 0.5, (img_w, img_h))

    R = np.eye(3)
    T = np.array([[baseline_mm], [0.0], [0.0]])

    # Dla identycznych kamer bez obrotu: R1=R2=I, P1=K|0, P2=K|[-f*B,0,0]
    R1 = R2 = np.eye(3)
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K, np.array([[-baseline_mm * focal_mm], [0], [0]])])

    # Macierz Q dla poziomej konfiguracji stereo (wyprowadzenie analityczne).
    # Srodkowe dwa wiersze przeliczaja wspolrzedne pikselowe na katowe,
    # trzeci wiersz koduje ogniskowa, czwarty - odwrotnosc bazy (1/B).
    # Znak -1/baseline daje Z > 0 gdy dysparycja > 0 i kamera prawa jest
    # przesunieta w prawo wzgledem lewej (T[0] > 0, konwencja OpenCV).
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

    # Tlo na glebokosci 3000mm - dalej niz wszystkie testowe pudla,
    # reprezentuje "podloge" lub daleki plan sceny
    bg_depth = 3000.0
    left_img  = np.full((img_h, img_w, 3), 40, dtype=np.uint8)
    right_img = np.full((img_h, img_w, 3), 40, dtype=np.uint8)
    depth_gt  = np.full((img_h, img_w), bg_depth, dtype=np.float32)

    # Dysparycja tla obliczana ze wzoru: d = f * B / Z (rownolegla konfiguracja stereo)
    bg_disp = focal_mm * baseline_mm / bg_depth

    for x0, y0, w, h, z_mm, color in boxes:
        disp_mm = focal_mm * baseline_mm / z_mm
        # Prawy obraz jest przesuniany o dysparycje w lewo (efekt paralaksy)
        shift_px = int(round(disp_mm))  # przesuniecie w pikselach dla prawego obrazu

        _render_box(depth_gt,  left_img,  x0,            y0, w, h, z_mm,   color)
        _render_box(depth_gt,  right_img, x0 - shift_px, y0, w, h, z_mm,   color)

    # Szum Gaussa symuluje realistyczne zaklocenia sensora kamery mobilnej.
    # Uzywamy stalego ziarna dla powtarzalnosci wynikow testow.
    noise_l = np.random.RandomState(0).randint(-8, 8, left_img.shape).astype(np.int16)
    noise_r = np.random.RandomState(1).randint(-8, 8, right_img.shape).astype(np.int16)
    left_img  = np.clip(left_img.astype(np.int16)  + noise_l, 0, 255).astype(np.uint8)
    right_img = np.clip(right_img.astype(np.int16) + noise_r, 0, 255).astype(np.uint8)

    log.info("Scena syntetyczna: %dx%d px, %d pudel, baza=%.0f mm",
             img_w, img_h, len(boxes), baseline_mm)
    return left_img, right_img, depth_gt, stereo


# ===========================================================================
# METRYKI I RAPORT KONCOWY
# ===========================================================================

def compute_depth_error(
    depth_est: np.ndarray,
    depth_gt: np.ndarray,
    max_depth_mm: float = 4000.0,
) -> dict:
    """Oblicza metryki bledu mapy glebokosci wzgledem ground-truth.

    Dostepne tylko w trybie syntetycznym, gdzie znamy dokladna glebokos
    kazdego piksela. Metryki trafiaja do raportu koncowego i pliku report.txt.

    Args:
        depth_est:    oszacowana mapa glebokosci [mm] z disparity_to_depth()
        depth_gt:     mapa glebokosci ground-truth [mm] z generate_synthetic_scene()
        max_depth_mm: gorny prog - piksele powyzej sa ignorowane

    Returns:
        Slownik z metrykami: MAE, MedAE, RMSE, blad wzgledny, pokrycie
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
    """Drukuje podsumowanie wynikow pipeline na standardowe wyjscie.

    Zawiera parametry kalibracji stereo, metryki bledu glebokosci
    (tylko tryb syntetyczny) oraz opis metodologii i zrodel bledow.
    Wynik mozna przekleic bezposrednio do raportu projektowego.
    """
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
    # 1. Wczytanie / wygenerowanie pary stereo
    # ------------------------------------------------------------------
    if mode == "synthetic":
        log.info("=== TRYB SYNTETYCZNY ===")
        left_img, right_img, depth_gt, stereo = generate_synthetic_scene()
        # W trybie syntetycznym mamy ground-truth glebokosci - mozemy policzyc blad
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
        # W trybie rzeczywistym nie mamy ground-truth - metryki bledu niedostepne
        metrics_possible = False
        log.info("Wczytano: %s | %s", left_path, right_path)

    # Zapis kopii obrazow wejsciowych dla pozniejszego podgladu i debugowania
    cv2.imwrite(str(out / "input_left.png"),  left_img)
    cv2.imwrite(str(out / "input_right.png"), right_img)

    # ------------------------------------------------------------------
    # 2. Rektyfikacja - wyrownanie linii epipolarnych obu obrazow
    # ------------------------------------------------------------------
    log.info("--- Rektyfikacja ---")
    # Nie wymuszamy rozmiaru wejscia - rectify_pair uzywa rozdzielczosci kalibracji
    # (stereo.left.image_size) i w razie potrzeby sam dopasowuje obrazy, dzieki czemu
    # macierz Q pozostaje poprawna. W trybie syntetycznym rozmiary i tak sie zgadzaja.
    left_rect, right_rect = rectify_pair(stereo, left_img, right_img)

    cv2.imwrite(str(out / "left_rect.png"),  left_rect)
    cv2.imwrite(str(out / "right_rect.png"), right_rect)
    cv2.imwrite(str(out / "epipolar_check.png"),
                draw_epipolar_check(left_rect, right_rect))

    # ------------------------------------------------------------------
    # 3. Mapa dysparycji - SGBM
    # ------------------------------------------------------------------
    log.info("--- Mapa dysparycji (SGBM) ---")
    # Parametry dobrane dla typowej sceny: obiekty w odleglosci 0.5-3m,
    # baza ~120mm, telefony z aparatem o rozdzielczosci Full HD
    cfg = SGBMConfig(
        num_disparities=config.SGBM_NUM_DISPARITIES,
        block_size=7,
        uniqueness=10,
        speckle_window=100,
    )
    disp = compute_disparity(left_rect, right_rect, cfg)

    np.save(str(out / "disparity.npy"), disp)
    cv2.imwrite(str(out / "disparity_color.png"), colormap_disparity(disp))

    # ------------------------------------------------------------------
    # 4. Mapa glebokosci - reprojekcja dysparycji przez macierz Q
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
    # 5-7. Detekcja palety, segmentacja obiektu, pomiar wymiarow
    # Blok try/except: nieudana detekcja palety nie przerywa pipeline -
    # pozostale wyniki (chmura, mapy) sa i tak zapisane i uzyteczne.
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
        meas = measure_object(xyz, pallet_result, noise_floor_mm=config.NOISE_FLOOR_MM)
        validation = validate_measurement(meas)
        report_text = generate_report(meas, validation)
        print(report_text)

        with open(str(out / "measurement_report.txt"), "w") as f:
            f.write(report_text)
        log.info("Raport zapisany: %s", out / "measurement_report.txt")

        _save_pallet_debug(
            out / "debug" / "pallet_debug.png",
            pallet_result,
            meas,
            noise_floor_mm=config.NOISE_FLOOR_MM,
        )

    except (RuntimeError, ValueError) as e:
        log.warning("Detekcja palety / pomiar nieudane: %s", e)

    # ------------------------------------------------------------------
    # 8. Metryki bledu glebokosci (tylko tryb syntetyczny) i raport koncowy
    # ------------------------------------------------------------------
    metrics = {}
    if metrics_possible and depth_gt is not None:
        # Porownujemy szacowana glebokos ze znana wartoscia ground-truth
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