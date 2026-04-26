# test_measurement.py
"""Testy jednostkowe dla modulow pallet.py i measurement.py.

Wszystkie testy uzywaja syntetycznych chmur punktow wygenerowanych wprost
(bez SGBM), co pozwala testowac geometrie niezaleznie od jakosci mapy dysparycji.

Konwencja ukladu kamery:
  - os Z wskazuje "od kamery" - wieksze Z = dalej od kamery
  - paleta lezy na plaszczynie Z=pallet_z, jej normalna wskazuje ku kamerze (Z maleje)
  - obiekty sa blizej kamery niz paleta: Z_obj < pallet_z
"""
import numpy as np
import pytest

from pallet import (
    PlaneModel,
    PalletDetectionResult,
    detect_pallet_plane,
    transform_to_pallet_frame,
    filter_roi,
    detect_pallet,
)
from measurement import (
    BoundingBox,
    MeasurementResult,
    ValidationReport,
    segment_object,
    compute_bounding_box,
    extract_3d_contour,
    measure_object,
    validate_measurement,
    generate_report,
)


# ===========================================================================
# syntetyczna scena paleta + pudelko
# ===========================================================================

def _make_pallet_scene(
    box_dims: tuple = (300.0, 200.0, 150.0),
    box_center_xy: tuple = (0.0, 0.0),
    pallet_z: float = 1500.0,
    n_pallet: int = 5000,
    n_object: int = 2000,
    noise_mm: float = 2.0,
    rng_seed: int = 42,
    tilt_deg: float = 0.0,
) -> tuple[np.ndarray, BoundingBox]:
    """Generuje czystą chmure punktow: plaszczyzna palety + pudelko nad nia.

    Uklad kamery: os Z od kamery (wieksze Z = dalej).
    Paleta: Z = pallet_z ± noise_mm, XY w [-600,600] x [-400,400].
    Pudelko: centrum XY = box_center_xy, Z w [pallet_z-box_h, pallet_z].

    Args:
        box_dims:     (szerokosc, dlugosc, wysokosc) pudla [mm]
        box_center_xy: (cx, cy) srodek pudla w ukladzie XY [mm]
        pallet_z:     glebokos palety [mm]
        n_pallet:     liczba punktow palety
        n_object:     liczba punktow pudelka
        noise_mm:     odchylenie std szumu Z palety [mm]
        rng_seed:     ziarno RNG
        tilt_deg:     kat nachylenia plaszczyzny palety [stopnie] wokol osi X

    Returns:
        (xyz, expected_bbox) gdzie expected_bbox to nominalne wymiary pudla
    """
    rng = np.random.RandomState(rng_seed)
    bw, bl, bh = box_dims
    cx, cy = box_center_xy

    # Punkty palety - plaszczyzna Z=pallet_z, XY w [-600,600] x [-400,400]
    px = rng.uniform(-600, 600, n_pallet).astype(np.float32)
    py = rng.uniform(-400, 400, n_pallet).astype(np.float32)
    pz = np.full(n_pallet, pallet_z, dtype=np.float32) + \
         rng.normal(0, noise_mm, n_pallet).astype(np.float32)
    pallet_pts = np.column_stack([px, py, pz])

    # Punkty pudelka - rowniez w ukladzie kamery
    # Pudelko jest blizej kamery: Z w [pallet_z - bh, pallet_z - noise_floor]
    ox = rng.uniform(cx - bw/2, cx + bw/2, n_object).astype(np.float32)
    oy = rng.uniform(cy - bl/2, cy + bl/2, n_object).astype(np.float32)
    oz = rng.uniform(pallet_z - bh, pallet_z - 25.0, n_object).astype(np.float32)
    object_pts = np.column_stack([ox, oy, oz])

    xyz = np.vstack([pallet_pts, object_pts])

    # Opcjonalne nachylenie (obrót wokol osi X o tilt_deg)
    if tilt_deg != 0.0:
        theta = np.radians(tilt_deg)
        Rx = np.array([
            [1, 0,             0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)],
        ], dtype=np.float32)
        xyz = (Rx @ xyz.T).T

    # Oczekiwany bbox w ukladzie palety (przed obrotem - bo obrót jest maly)
    # Wyskokosc to bh minus noise_floor (25 mm)
    expected_bbox = BoundingBox(
        x_min=cx - bw/2, x_max=cx + bw/2,
        y_min=cy - bl/2, y_max=cy + bl/2,
        z_min=25.0, z_max=bh,
        width=bw, length=bl, height=bh - 25.0,
    )

    return xyz, expected_bbox


# ===========================================================================
# TestRANSACPlaneDetection
# ===========================================================================

class TestRANSACPlaneDetection:

    def test_finds_horizontal_plane(self):
        xyz, _ = _make_pallet_scene(noise_mm=1.0)
        plane = detect_pallet_plane(xyz, n_iterations=500, distance_threshold=10.0, min_inliers=100)

        # Normalna powinna byc w przyblizeniu [0,0,-1] (wskazuje ku kamerze)
        angle = np.arccos(np.clip(abs(np.dot(plane.normal, [0, 0, -1])), 0, 1))
        assert np.degrees(angle) < 5.0, f"Normalna odchylona o {np.degrees(angle):.1f} stopni"
        assert plane.rms_residual < 5.0, f"RMS={plane.rms_residual:.2f} mm"

    def test_handles_plane_with_higher_noise(self):
        xyz, _ = _make_pallet_scene(noise_mm=5.0)
        plane = detect_pallet_plane(xyz, n_iterations=500, distance_threshold=15.0, min_inliers=100)

        angle = np.arccos(np.clip(abs(np.dot(plane.normal, [0, 0, -1])), 0, 1))
        assert np.degrees(angle) < 10.0

    def test_inlier_count_close_to_n_pallet(self):
        n_pallet = 3000
        xyz, _ = _make_pallet_scene(n_pallet=n_pallet, n_object=500, noise_mm=1.0)
        plane = detect_pallet_plane(xyz, n_iterations=500, distance_threshold=5.0, min_inliers=100)

        # Inliery powinny byc bliskie n_pallet (paleta to dominujaca plaszczyzna)
        assert plane.inlier_mask.sum() >= int(0.8 * n_pallet), \
            f"Zbyt malo inlierow: {plane.inlier_mask.sum()} < {int(0.8*n_pallet)}"

    def test_raises_on_too_few_points(self):
        with pytest.raises(ValueError, match="Za malo"):
            detect_pallet_plane(np.zeros((2, 3)))

    def test_raises_when_no_dominant_plane(self):
        rng = np.random.RandomState(0)
        xyz_random = rng.uniform(-1000, 1000, (500, 3)).astype(np.float32)
        with pytest.raises(RuntimeError, match="Nie znaleziono"):
            detect_pallet_plane(xyz_random, n_iterations=200, min_inliers=400)

    def test_inlier_mask_shape_matches_input(self):
        xyz, _ = _make_pallet_scene()
        plane = detect_pallet_plane(xyz, min_inliers=50)
        assert plane.inlier_mask.shape == (len(xyz),)

    def test_normal_is_unit_vector(self):
        xyz, _ = _make_pallet_scene()
        plane = detect_pallet_plane(xyz, min_inliers=50)
        assert abs(np.linalg.norm(plane.normal) - 1.0) < 1e-6


# ===========================================================================
# TestPalletTransform
# ===========================================================================

class TestPalletTransform:

    @pytest.fixture(scope="class")
    def scene(self):
        xyz, _ = _make_pallet_scene(noise_mm=1.0, n_pallet=4000)
        plane = detect_pallet_plane(xyz, n_iterations=500, distance_threshold=8.0, min_inliers=100)
        return xyz, plane

    def test_pallet_surface_near_zero_height(self, scene):
        xyz, plane = scene
        xyz_pallet, _, _ = transform_to_pallet_frame(xyz, plane)
        # Inliery powinny miec Z_pallet bliskie 0
        inlier_z = xyz_pallet[plane.inlier_mask, 2]
        assert abs(inlier_z.mean()) < 5.0, f"Srednia Z inlierow: {inlier_z.mean():.2f} mm"
        assert inlier_z.std() < 10.0

    def test_object_has_positive_height(self, scene):
        xyz, plane = scene
        xyz_pallet, _, _ = transform_to_pallet_frame(xyz, plane)
        non_inlier_z = xyz_pallet[~plane.inlier_mask, 2]
        # Wiekszosc punktow nie-inlierow (pudelko) powinna miec Z > 0
        assert (non_inlier_z > 0).mean() > 0.7

    def test_distance_preservation(self, scene):
        xyz, plane = scene
        xyz_pallet, _, _ = transform_to_pallet_frame(xyz, plane)
        idx = np.random.RandomState(7).choice(len(xyz), 50, replace=False)
        for i in range(0, len(idx) - 1, 2):
            d_cam = np.linalg.norm(xyz[idx[i]] - xyz[idx[i+1]])
            d_pal = np.linalg.norm(xyz_pallet[idx[i]] - xyz_pallet[idx[i+1]])
            assert abs(d_cam - d_pal) < 0.01, f"Odleglosc sie zmienila: {d_cam:.3f} vs {d_pal:.3f}"

    def test_roi_filter_excludes_far_points(self):
        rng = np.random.RandomState(0)
        # Punkty z XY wyraznie poza ROI
        xyz_far = np.column_stack([
            rng.uniform(700, 1000, 100),
            rng.uniform(500, 800, 100),
            np.zeros(100),
        ]).astype(np.float32)
        mask = filter_roi(xyz_far)
        assert mask.sum() == 0, "Punkty poza ROI nie zostaly wykluczone"

    def test_roi_filter_includes_interior_points(self):
        rng = np.random.RandomState(0)
        # Punkty wyraznie wewnatrz ROI
        xyz_in = np.column_stack([
            rng.uniform(-500, 500, 100),
            rng.uniform(-300, 300, 100),
            np.zeros(100),
        ]).astype(np.float32)
        mask = filter_roi(xyz_in)
        assert mask.sum() == 100

    def test_roi_boundary_exact(self):
        pts = np.array([
            [600.0, 400.0, 0.0],   # na granicy - wewnatrz
            [600.1, 400.1, 0.0],   # tuż za granica - na zewnatrz
        ], dtype=np.float32)
        mask = filter_roi(pts)
        assert mask[0] == True
        assert mask[1] == False


# ===========================================================================
# TestObjectSegmentation
# ===========================================================================

class TestObjectSegmentation:

    def test_segment_separates_object(self):
        xyz, _ = _make_pallet_scene(n_pallet=3000, n_object=1000, noise_mm=1.0)
        plane = detect_pallet_plane(xyz, n_iterations=500, distance_threshold=8.0, min_inliers=100)
        xyz_pallet, _, _ = transform_to_pallet_frame(xyz, plane)
        roi_mask = filter_roi(xyz_pallet)

        xyz_roi = xyz_pallet[roi_mask]
        obj_mask = segment_object(xyz_roi, noise_floor_mm=20.0)

        # Pudelko ma ~1000 punktow, paleta ~3000; oczekujemy ze obj_mask lapie wiekszos pudla
        assert obj_mask.sum() > 200, f"Za malo punktow obiektu: {obj_mask.sum()}"

    def test_pallet_only_gives_no_object(self):
        rng = np.random.RandomState(0)
        # Czysta plaszczyzna bez obiektu
        n = 3000
        xyz_pallet_frame = np.column_stack([
            rng.uniform(-500, 500, n),
            rng.uniform(-300, 300, n),
            rng.normal(0, 3.0, n),  # szum Z ~3 mm wokol 0
        ]).astype(np.float32)

        obj_mask = segment_object(xyz_pallet_frame, noise_floor_mm=20.0)
        assert obj_mask.sum() == 0

    def test_custom_noise_floor_changes_result(self):
        rng = np.random.RandomState(1)
        n = 1000
        xyz = np.column_stack([
            rng.uniform(-100, 100, n),
            rng.uniform(-100, 100, n),
            rng.uniform(5, 50, n),  # Z w [5, 50]
        ]).astype(np.float32)

        # Z noise_floor=20 lapie tylko punkty Z > 20
        mask_20 = segment_object(xyz, noise_floor_mm=20.0)
        # Z noise_floor=0 lapie wszystkie
        mask_0 = segment_object(xyz, noise_floor_mm=0.0)

        assert mask_0.sum() > mask_20.sum()


# ===========================================================================
# TestBoundingBox
# ===========================================================================

class TestBoundingBox:

    def test_known_geometry_width_length(self):
        bw, bl, bh = 300.0, 200.0, 150.0
        rng = np.random.RandomState(5)
        xyz = np.column_stack([
            rng.uniform(-bw/2, bw/2, 2000),
            rng.uniform(-bl/2, bl/2, 2000),
            rng.uniform(25.0, bh, 2000),
        ]).astype(np.float32)

        bbox = compute_bounding_box(xyz)

        assert abs(bbox.width - bw) < 5.0, f"Szerokosc: {bbox.width:.1f} vs {bw}"
        assert abs(bbox.length - bl) < 5.0, f"Dlugosc: {bbox.length:.1f} vs {bl}"
        assert bbox.height > 0

    def test_raises_on_empty_cloud(self):
        with pytest.raises(ValueError, match="Pusta"):
            compute_bounding_box(np.zeros((0, 3), dtype=np.float32))

    def test_all_dimensions_positive(self):
        rng = np.random.RandomState(0)
        xyz = rng.uniform(-100, 100, (500, 3)).astype(np.float32)
        bbox = compute_bounding_box(xyz)
        assert bbox.width > 0
        assert bbox.length > 0
        assert bbox.height > 0

    def test_single_point_gives_zero_dims(self):
        xyz = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        bbox = compute_bounding_box(xyz)
        assert bbox.width == 0.0
        assert bbox.length == 0.0
        assert bbox.height == 0.0


# ===========================================================================
# TestContour
# ===========================================================================

class TestContour:

    def test_contour_encloses_all_points(self):
        rng = np.random.RandomState(0)
        xyz = np.column_stack([
            rng.uniform(-100, 100, 500),
            rng.uniform(-80, 80, 500),
            rng.uniform(0, 50, 500),
        ]).astype(np.float32)

        hull = extract_3d_contour(xyz)
        assert hull.shape[1] == 2

        # Sprawdz ze hull otacza wszystkie punkty (wszystkie wewnatrz lub na granicy)
        pts_2d = xyz[:, :2].astype(np.float32)
        hull_int = hull.reshape(-1, 1, 2).astype(np.float32)
        for pt in pts_2d:
            dist = cv2.pointPolygonTest(hull_int, tuple(pt.tolist()), True)
            assert dist >= -1.0, f"Punkt {pt} na zewnatrz hull ({dist:.2f})"

    def test_too_few_points_returns_input(self):
        pts = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], dtype=np.float32)
        result = extract_3d_contour(pts)
        assert result.shape == (2, 2)

    def test_contour_bounding_rect_matches_bbox(self):
        bw, bl = 200.0, 150.0
        rng = np.random.RandomState(3)
        xyz = np.column_stack([
            rng.uniform(-bw/2, bw/2, 1000),
            rng.uniform(-bl/2, bl/2, 1000),
            rng.uniform(0, 50, 1000),
        ]).astype(np.float32)

        hull = extract_3d_contour(xyz)
        hull_x_span = hull[:, 0].max() - hull[:, 0].min()
        hull_y_span = hull[:, 1].max() - hull[:, 1].min()

        bbox = compute_bounding_box(xyz)
        assert abs(hull_x_span - bbox.width) < 2.0
        assert abs(hull_y_span - bbox.length) < 2.0


# ===========================================================================
# TestMeasureObject
# ===========================================================================

class TestMeasureObject:

    @pytest.fixture(scope="class")
    def measurement(self):
        xyz, expected_bbox = _make_pallet_scene(
            box_dims=(300.0, 200.0, 150.0),
            n_pallet=5000, n_object=2000, noise_mm=1.0,
        )
        pallet_result = detect_pallet(xyz, n_iterations=500, distance_threshold=8.0,
                                      min_inliers=100)
        meas = measure_object(xyz, pallet_result, noise_floor_mm=20.0)
        return meas, expected_bbox

    def test_returns_measurement_result_type(self, measurement):
        meas, _ = measurement
        assert isinstance(meas, MeasurementResult)

    def test_width_within_tolerance(self, measurement):
        meas, expected = measurement
        assert abs(meas.bbox.width - expected.width) < 15.0, \
            f"Szerokosc: {meas.bbox.width:.1f} vs {expected.width:.1f}"

    def test_length_within_tolerance(self, measurement):
        meas, expected = measurement
        assert abs(meas.bbox.length - expected.length) < 15.0, \
            f"Dlugosc: {meas.bbox.length:.1f} vs {expected.length:.1f}"

    def test_height_within_tolerance(self, measurement):
        meas, expected = measurement
        # Wysokosc jest mierzona nad noise_floor (25mm), wiec oczekujemy ~bh-25=125mm
        assert meas.bbox.height > 50.0
        assert meas.bbox.height < 200.0

    def test_raises_when_no_object(self):
        # Czysta paleta bez pudla
        rng = np.random.RandomState(0)
        n = 5000
        xyz = np.column_stack([
            rng.uniform(-500, 500, n),
            rng.uniform(-300, 300, n),
            np.full(n, 1500.0) + rng.normal(0, 2.0, n),
        ]).astype(np.float32)

        pallet_result = detect_pallet(xyz, n_iterations=300, distance_threshold=10.0,
                                      min_inliers=100)
        with pytest.raises(RuntimeError, match="Brak punktow"):
            measure_object(xyz, pallet_result, noise_floor_mm=20.0)


# ===========================================================================
# TestValidation
# ===========================================================================

class TestValidation:

    @pytest.fixture(scope="class")
    def good_measurement(self):
        xyz, _ = _make_pallet_scene(n_pallet=5000, n_object=2000, noise_mm=1.0)
        pallet_result = detect_pallet(xyz, n_iterations=500, distance_threshold=8.0,
                                      min_inliers=100)
        return measure_object(xyz, pallet_result, noise_floor_mm=20.0)

    def test_passes_on_good_measurement(self, good_measurement):
        validation = validate_measurement(good_measurement)
        assert validation.passed, f"Oczekiwano PASS, issues: {validation.issues}"

    def test_flags_high_rms(self, good_measurement):
        # Wstrzyknij zly RMS przez modyfikacje PlaneModel
        import dataclasses
        bad_plane = dataclasses.replace(good_measurement.pallet_result.plane, rms_residual=100.0)
        bad_pallet = dataclasses.replace(good_measurement.pallet_result, plane=bad_plane)
        bad_meas = dataclasses.replace(good_measurement, pallet_result=bad_pallet)

        validation = validate_measurement(bad_meas, max_pallet_rms_mm=30.0)
        assert not validation.passed
        assert any("RMS" in issue for issue in validation.issues)

    def test_flags_too_few_inliers(self, good_measurement):
        import dataclasses
        # Symulujemy malo inlierow przez zmiane n_pallet_inliers
        bad_meas = dataclasses.replace(good_measurement, n_pallet_inliers=10)

        validation = validate_measurement(bad_meas, min_inliers=100)
        assert not validation.passed
        assert any("inlier" in issue.lower() for issue in validation.issues)

    def test_flags_bad_height(self, good_measurement):
        import dataclasses
        bad_bbox = dataclasses.replace(
            good_measurement.bbox,
            z_min=0.0, z_max=3.0, height=3.0
        )
        bad_meas = dataclasses.replace(good_measurement, bbox=bad_bbox)

        validation = validate_measurement(bad_meas, min_height_mm=10.0)
        assert not validation.passed


# ===========================================================================
# TestReport
# ===========================================================================

class TestReport:

    @pytest.fixture(scope="class")
    def report_data(self):
        xyz, _ = _make_pallet_scene(n_pallet=4000, n_object=1500, noise_mm=1.0)
        pallet_result = detect_pallet(xyz, n_iterations=500, distance_threshold=8.0,
                                      min_inliers=100)
        meas = measure_object(xyz, pallet_result, noise_floor_mm=20.0)
        val = validate_measurement(meas)
        return meas, val

    def test_report_is_string(self, report_data):
        meas, val = report_data
        report = generate_report(meas, val)
        assert isinstance(report, str)

    def test_report_contains_mm(self, report_data):
        meas, val = report_data
        report = generate_report(meas, val)
        assert "mm" in report

    def test_report_contains_pass_or_fail(self, report_data):
        meas, val = report_data
        report = generate_report(meas, val)
        assert "PASS" in report or "FAIL" in report

    def test_report_contains_header(self, report_data):
        meas, val = report_data
        report = generate_report(meas, val)
        assert "RAPORT" in report

    def test_report_contains_dimensions(self, report_data):
        meas, val = report_data
        report = generate_report(meas, val)
        # Sprawdz ze raport zawiera sekcje z wymiarami
        assert "Szerokosc" in report
        assert "Dlugosc" in report
        assert "Wysokosc" in report


# ===========================================================================
# Import cv2 (potrzebny dla extract_3d_contour - pointPolygonTest)
# ===========================================================================
import cv2
