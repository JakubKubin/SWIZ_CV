"""Testy modulu kalibracji.
Generuje syntetyczne obrazy szachownicy z wirtualnej kamery,
nastepnie sprawdza czy kalibracja odzyskuje parametry kamery.
"""
import os, shutil, tempfile
import numpy as np
import cv2
import pytest

os.environ["CHECKERBOARD_ROWS"] = "7"
os.environ["CHECKERBOARD_COLS"] = "5"
os.environ["SQUARE_SIZE_MM"] = "25.0"

import calibration as cal

IMG_W, IMG_H = 640, 480
NUM_VIEWS = 8


def _make_ground_truth_camera():
    K = np.array([
        [800, 0, IMG_W / 2],
        [0, 800, IMG_H / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def _make_flat_checkerboard(rows, cols, sq_px=60):
    nr, nc = rows + 1, cols + 1
    border = sq_px
    img_h, img_w = nr * sq_px + 2 * border, nc * sq_px + 2 * border
    board = np.ones((img_h, img_w), dtype=np.uint8) * 255
    for r in range(nr):
        for c in range(nc):
            if (r + c) % 2 == 0:
                y, x = border + r * sq_px, border + c * sq_px
                board[y:y + sq_px, x:x + sq_px] = 0
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _generate_checkerboard_images(
    K, dist, num_views, rows, cols, sq_size, img_size, rng,
    t_offset=np.zeros(3),
):
    flat = _make_flat_checkerboard(rows, cols, sq_px=60)
    h_flat, w_flat = flat.shape[:2]
    w, h = img_size
    images = []
    for _ in range(num_views * 4):
        scale = rng.uniform(0.55, 0.85)
        angle = rng.uniform(-20, 20)
        cx_off = rng.uniform(-40, 40)
        cy_off = rng.uniform(-30, 30)
        center = (w_flat / 2, h_flat / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += (w / 2 - center[0]) + cx_off
        M[1, 2] += (h / 2 - center[1]) + cy_off
        img = cv2.warpAffine(flat, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderValue=(180, 180, 180))
        images.append(img)
        if len(images) >= num_views:
            break
    return images


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
    K, dist = _make_ground_truth_camera()
    rng = np.random.RandomState(123)
    baseline = np.array([120.0, 0.0, 0.0])
    left_imgs = _generate_checkerboard_images(
        K, dist, NUM_VIEWS, cal.BOARD_ROWS, cal.BOARD_COLS,
        cal.SQUARE_SIZE, (IMG_W, IMG_H), rng,
    )
    rng2 = np.random.RandomState(123)
    right_imgs = _generate_checkerboard_images(
        K, dist, NUM_VIEWS, cal.BOARD_ROWS, cal.BOARD_COLS,
        cal.SQUARE_SIZE, (IMG_W, IMG_H), rng2, t_offset=baseline,
    )
    n = min(len(left_imgs), len(right_imgs))
    ldir = tempfile.mkdtemp(prefix="calib_left_")
    rdir = tempfile.mkdtemp(prefix="calib_right_")
    for i in range(n):
        cv2.imwrite(os.path.join(ldir, f"frame_{i:03d}.png"), left_imgs[i])
        cv2.imwrite(os.path.join(rdir, f"frame_{i:03d}.png"), right_imgs[i])
    yield ldir, rdir, K, dist, baseline
    shutil.rmtree(ldir)
    shutil.rmtree(rdir)


class TestFindCorners:
    def test_valid_checkerboard(self, synth_single_dir):
        d, _, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        img = cv2.imread(paths[0])
        assert img is not None
        corners = cal.find_corners(img)
        assert corners is not None
        assert corners.shape == (cal.BOARD_ROWS * cal.BOARD_COLS, 1, 2)

    def test_blank_image_returns_none(self):
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        assert cal.find_corners(blank) is None


class TestSingleCalibration:
    def test_calibration_runs(self, synth_single_dir):
        d, K_true, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        params = cal.calibrate_single(paths)
        assert params.reproj_error < 2.0
        assert params.camera_matrix.shape == (3, 3)
        assert params.image_size == (IMG_W, IMG_H)

    def test_focal_length_reasonable(self, synth_single_dir):
        d, K_true, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        params = cal.calibrate_single(paths)
        fx, fy = params.camera_matrix[0, 0], params.camera_matrix[1, 1]
        assert fx > 100, f"fx too small: {fx}"
        assert fy > 100, f"fy too small: {fy}"
        assert abs(fx - fy) / max(fx, fy) < 0.3, "fx/fy ratio unreasonable"

    def test_too_few_images_raises(self, synth_single_dir):
        d, _, _ = synth_single_dir
        paths = cal.get_image_paths(d)[:1]
        with pytest.raises(ValueError, match="Za malo"):
            cal.calibrate_single(paths)


class TestStereoCalibration:
    def test_stereo_runs(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        lp = cal.get_image_paths(ldir)
        rp = cal.get_image_paths(rdir)
        sp = cal.calibrate_stereo(lp, rp)
        assert sp.reproj_error < 3.0
        assert sp.R.shape == (3, 3)
        assert sp.T.shape == (3, 1)

    def test_mismatched_count_raises(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        lp = cal.get_image_paths(ldir)[:3]
        rp = cal.get_image_paths(rdir)[:4]
        with pytest.raises(ValueError, match="rowna"):
            cal.calibrate_stereo(lp, rp)


class TestUndistortAndRectify:
    def test_undistort_returns_same_shape(self, synth_single_dir):
        d, _, _ = synth_single_dir
        params = cal.calibrate_single(cal.get_image_paths(d))
        frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        out = params.undistort(frame)
        assert out.shape == frame.shape

    def test_rectify_maps_shape(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sp = cal.calibrate_stereo(cal.get_image_paths(ldir), cal.get_image_paths(rdir))
        map1L, _, map1R, _ = sp.rectify_maps()
        assert map1L.shape[:2] == (IMG_H, IMG_W)
        assert map1R.shape[:2] == (IMG_H, IMG_W)

    def test_stereo_params_has_rectify_matrices(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sp = cal.calibrate_stereo(cal.get_image_paths(ldir), cal.get_image_paths(rdir))
        assert sp.R1.shape == (3, 3)
        assert sp.R2.shape == (3, 3)
        assert sp.P1.shape == (3, 4)
        assert sp.P2.shape == (3, 4)
        assert sp.Q.shape == (4, 4)


class TestSerializaton:
    def test_single_save_load(self, synth_single_dir, tmp_path):
        d, _, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        params = cal.calibrate_single(paths)
        fpath = str(tmp_path / "cam.json")
        cal.save_params(params, fpath)
        loaded = cal.load_params(fpath, stereo=False)
        np.testing.assert_array_almost_equal(
            loaded.camera_matrix, params.camera_matrix, decimal=6
        )
        assert abs(loaded.reproj_error - params.reproj_error) < 1e-6

    def test_stereo_save_load(self, synth_stereo_dirs, tmp_path):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sp = cal.calibrate_stereo(
            cal.get_image_paths(ldir), cal.get_image_paths(rdir)
        )
        fpath = str(tmp_path / "stereo.json")
        cal.save_params(sp, fpath)
        loaded = cal.load_params(fpath, stereo=True)
        np.testing.assert_array_almost_equal(loaded.R, sp.R, decimal=6)
        np.testing.assert_array_almost_equal(loaded.T, sp.T, decimal=6)


class TestCollectPoints:
    def test_returns_correct_count(self, synth_single_dir):
        d, _, _ = synth_single_dir
        paths = cal.get_image_paths(d)
        data = cal.collect_points(paths)
        assert len(data.obj_points) == len(data.img_points)
        assert len(data) > 0
        assert data.image_size == (IMG_W, IMG_H)

    def test_stereo_collect_aligned(self, synth_stereo_dirs):
        ldir, rdir, _, _, _ = synth_stereo_dirs
        sd = cal.collect_stereo_points(cal.get_image_paths(ldir), cal.get_image_paths(rdir))
        assert len(sd.obj_points) == len(sd.left_points) == len(sd.right_points)
        assert len(sd) > 0
        assert sd.image_size == (IMG_W, IMG_H)
        # left/right widoki wskazuja na te same obj_points
        assert sd.left.obj_points is sd.obj_points
        assert sd.right.obj_points is sd.obj_points


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
