# test_calibration.py
"""Testy modulu kalibracji.

Dwa rodzaje syntetycznych danych wejsciowych:
  1. Obrazy warpAffine (2D)  - testuja caly pipeline I/O + detekcji naroznikow.
     Nie sa spójne geometrycznie (nie odpowiadaja 3D kamerze) wiec nie nadaja sie
     do weryfikacji wartosci liczbowych, jedynie do sprawdzania ze kod sie wykonuje.

  2. Punkty z projectPoints (3D) - pomijaja rendering i find_corners;
     weryfikuja matematyke odzysku K, dystorsji i bazy stereo wzgledem GT.
"""
import os, shutil, tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

# Wyniki wizualne laduja tutaj – sprawdz je rcznie po uruchomieniu testow.
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

os.environ["CHECKERBOARD_ROWS"] = "7"
os.environ["CHECKERBOARD_COLS"] = "5"
os.environ["SQUARE_SIZE_MM"] = "25.0"

import calibration as cal

IMG_W, IMG_H = 640, 480
NUM_VIEWS = 8
# Poziome przesuniecie planszy dla prawej kamery [px].
# Ta sama macierz M dotyczy obu kamer; prawa kamera rozni sie jedynie x_shift_px.
# UWAGA: warpAffine nie modeluje 3D geometrii stereo - x_shift nie odpowiada
# fizycznej bazie. Sluzy tylko do weryfikacji ze pipeline I/O dziala.
BASELINE_PX = 40


# ---------------------------------------------------------------------------
# Generatory danych syntetycznych
# ---------------------------------------------------------------------------

def _make_ground_truth_camera():
    K = np.array([
        [800.0,   0.0, IMG_W / 2.0],
        [  0.0, 800.0, IMG_H / 2.0],
        [  0.0,   0.0,         1.0],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def _make_flat_checkerboard(rows, cols, sq_px=60):
    nr, nc = rows + 1, cols + 1
    border = sq_px
    img_h = nr * sq_px + 2 * border
    img_w = nc * sq_px + 2 * border
    board = np.ones((img_h, img_w), dtype=np.uint8) * 255
    for r in range(nr):
        for c in range(nc):
            if (r + c) % 2 == 0:
                board[border + r * sq_px:border + (r+1) * sq_px,
                      border + c * sq_px:border + (c+1) * sq_px] = 0
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _generate_checkerboard_images(
    K, dist, num_views, rows, cols, sq_size, img_size, rng,
    x_shift_px=0,
):
    """Generuje syntetyczne obrazy szachownicy przez warpAffine (symulacja 2D).

    x_shift_px: poziome przesuniecie planszy [px]. Uzywaj tego samego seeda
        dla lewej (x_shift_px=0) i prawej (x_shift_px=-BASELINE_PX), dzieki
        czemu obie kamery maja identyczna losowa pose - roznica to tylko shift.
    """
    flat = _make_flat_checkerboard(rows, cols, sq_px=60)
    h_flat, w_flat = flat.shape[:2]
    w, h = img_size
    images = []
    for _ in range(num_views * 4):
        scale = rng.uniform(0.55, 0.85)
        angle = rng.uniform(-20, 20)
        cx_off = rng.uniform(-40, 40)
        cy_off = rng.uniform(-30, 30)
        center = (w_flat / 2.0, h_flat / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += (w / 2.0 - center[0]) + cx_off + x_shift_px
        M[1, 2] += (h / 2.0 - center[1]) + cy_off
        img = cv2.warpAffine(flat, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderValue=(180, 180, 180))
        images.append(img)
        if len(images) >= num_views:
            break
    return images


def _make_projectpoints_calib_data(K, dist, rows, cols, sq_size, num_views, img_size, rng):
    """Generuje punkty kalibracyjne przez idealną 3D projekcje (cv2.projectPoints).

    Nie tworzy obrazow - punkty trafiaja bezposrednio do CalibrationData,
    z pominciem find_corners. Pozwala porownac odzysk K i dist z GT bez
    szumu detekcji naroznikow.
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * sq_size

    obj_pts, img_pts = [], []
    w, h = img_size

    for _ in range(num_views * 10):
        rvec = rng.uniform(-0.5, 0.5, (3, 1)).astype(np.float64)
        z = rng.uniform(350.0, 600.0)
        tx = rng.uniform(-sq_size * cols * 0.3, sq_size * cols * 0.3)
        ty = rng.uniform(-sq_size * rows * 0.3, sq_size * rows * 0.3)
        tvec = np.array([[tx], [ty], [z]], dtype=np.float64)

        pts, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 1, 2).astype(np.float32)
        xs, ys = pts[:, 0, 0], pts[:, 0, 1]
        if xs.min() > 10 and xs.max() < w - 10 and ys.min() > 10 and ys.max() < h - 10:
            obj_pts.append(objp)
            img_pts.append(pts)
        if len(obj_pts) >= num_views:
            break

    if len(obj_pts) < 3:
        raise RuntimeError(
            f"Za malo widokow ({len(obj_pts)}) - sprawdz parametry generatora"
        )
    return cal.CalibrationData(obj_pts, img_pts, img_size)


def _make_projectpoints_stereo_data(
    K, dist, rows, cols, sq_size, baseline_mm, num_views, img_size, rng
):
    """Generuje StereoCalibrationData przez 3D projekcje z pozioma baza stereo.

    Obie kamery widza ten sam wzorzec w tej samej chwili; prawa kamera jest
    przesunieta o baseline_mm w osi X wzgledem lewej (prawa baza stereo).
    Punkty trafiaja bezposrednio do StereoCalibrationData - brak renderowania.
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * sq_size

    # Prawa kamera jest w polozeniu [baseline_mm, 0, 0] w ukladzie lewej kamery.
    # Plansza widziana z prawej kamery: tvec_right = tvec_left - [baseline, 0, 0]^T
    T_stereo = np.array([[baseline_mm], [0.0], [0.0]], dtype=np.float64)

    obj_pts, left_pts, right_pts = [], [], []
    w, h = img_size

    for _ in range(num_views * 10):
        rvec = rng.uniform(-0.4, 0.4, (3, 1)).astype(np.float64)
        z = rng.uniform(400.0, 800.0)
        tx = rng.uniform(-sq_size * cols * 0.15, sq_size * cols * 0.15)
        ty = rng.uniform(-sq_size * rows * 0.15, sq_size * rows * 0.15)
        tvec_left = np.array([[tx], [ty], [z]], dtype=np.float64)
        tvec_right = tvec_left - T_stereo

        lp, _ = cv2.projectPoints(objp, rvec, tvec_left, K, dist)
        rp, _ = cv2.projectPoints(objp, rvec, tvec_right, K, dist)
        lp = lp.reshape(-1, 1, 2).astype(np.float32)
        rp = rp.reshape(-1, 1, 2).astype(np.float32)

        lx, ly = lp[:, 0, 0], lp[:, 0, 1]
        rx, ry = rp[:, 0, 0], rp[:, 0, 1]
        if (lx.min() > 10 and lx.max() < w - 10 and ly.min() > 10 and ly.max() < h - 10 and
                rx.min() > 10 and rx.max() < w - 10 and ry.min() > 10 and ry.max() < h - 10):
            obj_pts.append(objp)
            left_pts.append(lp)
            right_pts.append(rp)
        if len(obj_pts) >= num_views:
            break

    if len(obj_pts) < 3:
        raise RuntimeError(f"Za malo widokow stereo ({len(obj_pts)})")
    return cal.StereoCalibrationData(obj_pts, left_pts, right_pts, img_size)


def _save_visual(img: np.ndarray, subdir: str, name: str, visualize: bool) -> None:
    """Zapisuje obraz do test_output/<subdir>/<name>.png jesli --visualize jest aktywne."""
    if not visualize:
        return
    out_dir = TEST_OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    cv2.imwrite(str(path), img)
    print(f"\n[visualize] Zapisano: {path}")


def _draw_corners_on_image(img: np.ndarray, corners) -> np.ndarray:
    """Rysuje wykryte narozniki; czerwony napis gdy nie znaleziono."""
    out = img.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            out, (cal.BOARD_COLS, cal.BOARD_ROWS), corners, patternWasFound=True
        )
    else:
        cv2.putText(out, "BRAK NAROZNIKOW", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _side_by_side(left: np.ndarray, right: np.ndarray,
                  label_l: str = "", label_r: str = "") -> np.ndarray:
    """Laczy dwa obrazy poziomo z opcjonalnymi etykietami."""
    combined = np.hstack([left, right])
    if label_l:
        cv2.putText(combined, label_l, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    if label_r:
        cv2.putText(combined, label_r, (left.shape[1] + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return combined


def _draw_epipolar_lines(combined: np.ndarray, n_lines: int = 12) -> np.ndarray:
    """Rysuje poziome linie epipolarne na juz polaczonym obrazie stereo."""
    h = combined.shape[0]
    out = combined.copy()
    step = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * step
        cv2.line(out, (0, y), (out.shape[1], y), (0, 200, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Fixtures: obrazy warpAffine - pelny pipeline I/O + find_corners
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_single_dir():
    K, dist = _make_ground_truth_camera()
    rng = np.random.RandomState(42)
    imgs = _generate_checkerboard_images(
        K, dist, NUM_VIEWS, cal.BOARD_ROWS, cal.BOARD_COLS,
        cal.SQUARE_SIZE, (IMG_W, IMG_H), rng,
    )
    tmpdir = tempfile.mkdtemp(prefix="calib_test_")
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(tmpdir, f"frame_{i:03d}.png"), img)
    yield tmpdir, K, dist
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="module")
def synth_stereo_dirs():
    """Pary stereo z warpAffine + x_shift - do testow pelnego pipeline.

    UWAGA: warpAffine nie modeluje poprawnej geometrii 3D stereo, wiec
    reproj_error z stereoCalibrate bedzie nieistotny. Ten fixture testuje
    jedynie poprawnosc kodu (I/O, ksztalty macierzy, zapis/odczyt).
    Do testow precyzji uzyj synth_stereo_data.
    """
    K, dist = _make_ground_truth_camera()
    ldir = tempfile.mkdtemp(prefix="calib_left_")
    rdir = tempfile.mkdtemp(prefix="calib_right_")
    left_imgs = _generate_checkerboard_images(
        K, dist, NUM_VIEWS, cal.BOARD_ROWS, cal.BOARD_COLS,
        cal.SQUARE_SIZE, (IMG_W, IMG_H), np.random.RandomState(123),
        x_shift_px=0,
    )
    right_imgs = _generate_checkerboard_images(
        K, dist, NUM_VIEWS, cal.BOARD_ROWS, cal.BOARD_COLS,
        cal.SQUARE_SIZE, (IMG_W, IMG_H), np.random.RandomState(123),
        x_shift_px=-BASELINE_PX,
    )
    n = min(len(left_imgs), len(right_imgs))
    for i in range(n):
        cv2.imwrite(os.path.join(ldir, f"frame_{i:03d}.png"), left_imgs[i])
        cv2.imwrite(os.path.join(rdir, f"frame_{i:03d}.png"), right_imgs[i])
    yield ldir, rdir, K, dist, BASELINE_PX
    shutil.rmtree(ldir)
    shutil.rmtree(rdir)


# ---------------------------------------------------------------------------
# Fixtures: punkty 3D - weryfikacja precyzyjna K, dystorsji i bazy stereo
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def precise_calib_data():
    """Idealna projekcja 3D bez dystorsji - do weryfikacji precyzji K (GT)."""
    K, dist = _make_ground_truth_camera()
    data = _make_projectpoints_calib_data(
        K, dist, cal.BOARD_ROWS, cal.BOARD_COLS, cal.SQUARE_SIZE,
        num_views=15, img_size=(IMG_W, IMG_H), rng=np.random.RandomState(77),
    )
    return data, K


@pytest.fixture(scope="module")
def distorted_calib_data():
    """Projekcja 3D z niezerowa dystorsja beczkowata (k1 > 0)."""
    K = np.array([
        [800.0,   0.0, IMG_W / 2.0],
        [  0.0, 800.0, IMG_H / 2.0],
        [  0.0,   0.0,         1.0],
    ], dtype=np.float64)
    dist_true = np.array([0.15, -0.08, 0.0, 0.0, 0.0], dtype=np.float64)
    data = _make_projectpoints_calib_data(
        K, dist_true, cal.BOARD_ROWS, cal.BOARD_COLS, cal.SQUARE_SIZE,
        num_views=15, img_size=(IMG_W, IMG_H), rng=np.random.RandomState(99),
    )
    return data, K, dist_true


@pytest.fixture(scope="module")
def synth_stereo_data():
    """Stereo 3D projectPoints z pozioma baza - do weryfikacji kierunku T."""
    K, dist = _make_ground_truth_camera()
    BASELINE_MM = 100.0
    data = _make_projectpoints_stereo_data(
        K, dist, cal.BOARD_ROWS, cal.BOARD_COLS, cal.SQUARE_SIZE,
        baseline_mm=BASELINE_MM, num_views=12, img_size=(IMG_W, IMG_H),
        rng=np.random.RandomState(456),
    )
    return data, K, BASELINE_MM


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

class TestFindCorners:
    def test_valid_checkerboard(self, synth_single_dir, visualize):
        """Sprawdza detekcje naroznikow na wszystkich syntetycznych klatkach.

        Przy --visualize zapisuje kazda klatke z narysowanymi naroznikami
        (lub napisem BRAK NAROZNIKOW) do test_output/corners/.
        """
        d, _, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        last_corners = None
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            assert img is not None, f"Nie mozna wczytac: {path}"
            corners = cal.find_corners(img)
            _save_visual(_draw_corners_on_image(img, corners),
                         "corners", f"frame_{i:03d}", visualize)
            assert corners is not None, f"Brak naroznikow: {path}"
            last_corners = corners
        assert last_corners is not None
        assert last_corners.shape == (cal.BOARD_ROWS * cal.BOARD_COLS, 1, 2)

    def test_blank_image_returns_none(self):
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        assert cal.find_corners(blank) is None


class TestSingleCalibration:
    def test_calibration_runs(self, synth_single_dir):
        d, _, _ = synth_single_dir
        params = cal.calibrate_single(cal.get_image_paths(d))
        assert params.reproj_error < 2.0
        assert params.camera_matrix.shape == (3, 3)
        assert params.image_size == (IMG_W, IMG_H)

    def test_focal_length_reasonable(self, synth_single_dir):
        d, _, _ = synth_single_dir
        params = cal.calibrate_single(cal.get_image_paths(d))
        fx, fy = params.camera_matrix[0, 0], params.camera_matrix[1, 1]
        assert fx > 100, f"fx too small: {fx}"
        assert fy > 100, f"fy too small: {fy}"
        assert abs(fx - fy) / max(fx, fy) < 0.3, "fx/fy ratio unreasonable"

    def test_camera_matrix_precision(self, precise_calib_data):
        """Weryfikuje precyzje odzysku K na idealnych danych 3D (GT).

        Dane z projectPoints + brak szumu → blad fx/fy < 5%, cx/cy < 5%.
        Nie mozna tego sprawdzic na obrazach warpAffine, bo nie koresponduja
        z zadnym modelem 3D kamery.
        """
        data, K_true = precise_calib_data
        params = cal._calibrate_from_data(data)
        diff = np.abs(params.camera_matrix - K_true)
        assert diff[0, 0] < K_true[0, 0] * 0.05, f"Blad fx: {diff[0,0]:.2f} px"
        assert diff[1, 1] < K_true[1, 1] * 0.05, f"Blad fy: {diff[1,1]:.2f} px"
        assert diff[0, 2] < IMG_W * 0.05,         f"Blad cx: {diff[0,2]:.2f} px"
        assert diff[1, 2] < IMG_H * 0.05,         f"Blad cy: {diff[1,2]:.2f} px"

    def test_too_few_images_raises(self, synth_single_dir):
        d, _, _ = synth_single_dir
        with pytest.raises(ValueError, match="Za malo"):
            cal.calibrate_single(cal.get_image_paths(d)[:1])


class TestDistortion:
    def test_nonzero_distortion_recovered(self, distorted_calib_data):
        """Kalibracja powinna wykryc niezerowa dystorsje beczkowata (k1 > 0)."""
        data, _, dist_true = distorted_calib_data
        params = cal._calibrate_from_data(data)
        k1_est = float(params.dist_coeffs.flat[0])
        k1_true = float(dist_true[0])
        assert np.sign(k1_est) == np.sign(k1_true), (
            f"Zly znak k1: est={k1_est:.4f}, true={k1_true:.4f}"
        )
        assert abs(k1_est - k1_true) < abs(k1_true) * 0.4, (
            f"k1 poza 40%% tolerancja: est={k1_est:.4f}, true={k1_true:.4f}"
        )

    def test_zero_distortion_near_zero(self, precise_calib_data):
        """Przy danych bez dystorsji wszystkie wspolczynniki dist powinny byc ~ 0."""
        data, _ = precise_calib_data
        params = cal._calibrate_from_data(data)
        assert np.all(np.abs(params.dist_coeffs) < 0.05), (
            f"Oczekiwano dist~0, otrzymano: {params.dist_coeffs.flatten()}"
        )


class TestStereoCalibration:
    def test_stereo_pipeline_runs(self, synth_stereo_dirs):
        """Weryfikuje ze caly pipeline I/O → find_corners → calibrate_stereo dziala.

        Nie sprawdza wartosci reproj_error - warpAffine nie modeluje geometrii 3D.
        """
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sp = cal.calibrate_stereo(cal.get_image_paths(ldir), cal.get_image_paths(rdir))
        assert sp.R.shape == (3, 3)
        assert sp.T.shape == (3, 1)
        assert sp.R1.shape == (3, 3)
        assert sp.Q.shape == (4, 4)

    def test_stereo_translation_accuracy(self, synth_stereo_data):
        """Weryfikuje ze odzyskane T dominuje w osi X (pozioma baza stereo).

        Uzywa danych 3D (projectPoints) z known baseline, wiec mozna
        sprawdzic i kierunek, i przyblizony rozmiar translacji.
        """
        data, _, baseline_mm = synth_stereo_data
        left_cam = cal._calibrate_from_data(data.left)
        right_cam = cal._calibrate_from_data(data.right)
        _, _, _, _, _, _, T, _, _ = cv2.stereoCalibrate(
            data.obj_points, data.left_points, data.right_points,
            left_cam.camera_matrix, left_cam.dist_coeffs,
            right_cam.camera_matrix, right_cam.dist_coeffs,
            data.image_size, criteria=cal.CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC,
        )
        t = T.flatten()
        assert np.argmax(np.abs(t)) == 0, (
            f"Translacja powinna dominowac w osi X, otrzymano T={t}"
        )
        # Wielkosc T powinna odpowiadac baseline (w jednostkach SQUARE_SIZE_MM)
        assert abs(abs(t[0]) - baseline_mm) < baseline_mm * 0.15, (
            f"|T[0]|={abs(t[0]):.1f} mm, oczekiwano ~{baseline_mm} mm"
        )

    def test_mismatched_count_raises(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        with pytest.raises(ValueError, match="rowna"):
            cal.calibrate_stereo(
                cal.get_image_paths(ldir)[:3],
                cal.get_image_paths(rdir)[:4],
            )


class TestUndistortAndRectify:
    def test_undistort_returns_same_shape(self, synth_single_dir, visualize):
        d, _, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        params = cal.calibrate_single(paths)
        frame = cv2.imread(paths[0])
        assert frame is not None
        undistorted = params.undistort(frame)
        assert undistorted.shape == frame.shape
        _save_visual(
            _side_by_side(frame, undistorted, "Oryginal", "Po undistort"),
            "undistort", "before_after", visualize,
        )

    def test_rectify_maps_shape(self, synth_stereo_dirs, visualize):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        lpaths = cal.get_image_paths(ldir)
        rpaths = cal.get_image_paths(rdir)
        sp = cal.calibrate_stereo(lpaths, rpaths)
        map1L, map2L, map1R, map2R = sp.rectify_maps()
        assert map1L.shape[:2] == (IMG_H, IMG_W)
        assert map1R.shape[:2] == (IMG_H, IMG_W)
        # Rektyfikacja pierwszej pary + linie epipolarne – po rektyfikacji
        # odpowiadajace sobie punkty powinny lezec na tej samej linii poziomej.
        l_img = cv2.imread(lpaths[0])
        r_img = cv2.imread(rpaths[0])
        assert l_img is not None and r_img is not None
        l_rect = cv2.remap(l_img, map1L, map2L, cv2.INTER_LINEAR)
        r_rect = cv2.remap(r_img, map1R, map2R, cv2.INTER_LINEAR)
        combined = _side_by_side(l_rect, r_rect, "Lewa (rect)", "Prawa (rect)")
        _save_visual(_draw_epipolar_lines(combined), "rectify",
                     "pair_000_epipolar", visualize)

    def test_stereo_params_has_rectify_matrices(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sp = cal.calibrate_stereo(cal.get_image_paths(ldir), cal.get_image_paths(rdir))
        assert sp.R1.shape == (3, 3)
        assert sp.R2.shape == (3, 3)
        assert sp.P1.shape == (3, 4)
        assert sp.P2.shape == (3, 4)
        assert sp.Q.shape == (4, 4)


class TestSerialization:
    def test_single_save_load(self, synth_single_dir, tmp_path):
        d, _, _ = synth_single_dir
        params = cal.calibrate_single(cal.get_image_paths(d))
        fpath = str(tmp_path / "cam.json")
        cal.save_params(params, fpath)
        loaded = cal.load_params(fpath, stereo=False)
        np.testing.assert_array_almost_equal(
            loaded.camera_matrix, params.camera_matrix, decimal=6
        )
        assert abs(loaded.reproj_error - params.reproj_error) < 1e-6

    def test_stereo_save_load(self, synth_stereo_dirs, tmp_path):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sp = cal.calibrate_stereo(cal.get_image_paths(ldir), cal.get_image_paths(rdir))
        fpath = str(tmp_path / "stereo.json")
        cal.save_params(sp, fpath)
        loaded = cal.load_params(fpath, stereo=True)
        np.testing.assert_array_almost_equal(loaded.R,  sp.R,  decimal=6)
        np.testing.assert_array_almost_equal(loaded.T,  sp.T,  decimal=6)
        np.testing.assert_array_almost_equal(loaded.R1, sp.R1, decimal=6)
        np.testing.assert_array_almost_equal(loaded.Q,  sp.Q,  decimal=6)


class TestCollectPoints:
    def test_returns_correct_count(self, synth_single_dir):
        d, _, _ = synth_single_dir
        data = cal.collect_points(cal.get_image_paths(d))
        assert len(data.obj_points) == len(data.img_points)
        assert len(data) > 0
        assert data.image_size == (IMG_W, IMG_H)

    def test_stereo_collect_aligned(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sd = cal.collect_stereo_points(
            cal.get_image_paths(ldir), cal.get_image_paths(rdir),
        )
        assert len(sd.obj_points) == len(sd.left_points) == len(sd.right_points)
        assert len(sd) > 0
        assert sd.image_size == (IMG_W, IMG_H)
        # .left i .right wskazuja na te same obj_points - brak kopii
        assert sd.left.obj_points is sd.obj_points
        assert sd.right.obj_points is sd.obj_points


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
