"""Modul kalibracji kamer - metoda Zhanga (OpenCV).
Obsluguje kalibracje pojedynczej kamery oraz kalibracje stereo.
"""
import os, json, glob, logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, overload, Literal
import numpy as np
import cv2

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Aliasy dla czytelnosci (wartosci pochodza z config, ktory czyta env)
BOARD_ROWS = config.BOARD_ROWS
BOARD_COLS = config.BOARD_COLS
SQUARE_SIZE = config.SQUARE_SIZE_MM
CALIB_DIR = config.CALIB_DIR
CALIB_OUT = config.CALIB_OUTPUT
CRITERIA = config.TERM_CRITERIA


# ---------------------------------------------------------------------------
# Struktury danych dla wykrytych punktow kalibracyjnych
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Zebrane punkty kalibracyjne dla pojedynczej kamery."""
    obj_points: list   # list[np.ndarray shape (N,3)] - wspolrzedne 3D wzorca
    img_points: list   # list[np.ndarray shape (N,1,2)] - punkty 2D na obrazie
    image_size: tuple[int, int]  # (width, height)

    def __len__(self) -> int:
        return len(self.obj_points)


@dataclass
class StereoCalibrationData:
    """Zebrane punkty kalibracyjne dla pary stereo.
    Zawiera wylacznie pary, gdzie obie kamery wykryly wzorzec."""
    obj_points: list
    left_points: list
    right_points: list
    image_size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.obj_points)

    @property
    def left(self) -> CalibrationData:
        """Dane lewej kamery - gotowe do przekazania do _calibrate_from_data."""
        return CalibrationData(self.obj_points, self.left_points, self.image_size)

    @property
    def right(self) -> CalibrationData:
        """Dane prawej kamery - gotowe do przekazania do _calibrate_from_data."""
        return CalibrationData(self.obj_points, self.right_points, self.image_size)


# ---------------------------------------------------------------------------
# Parametry kamer
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    camera_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    reproj_error: float = 0.0
    image_size: tuple = (0, 0)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Zwraca obraz skorygowany o dystorsje obiektywu."""
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def to_dict(self):
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reproj_error": self.reproj_error,
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            camera_matrix=np.array(d["camera_matrix"]),
            dist_coeffs=np.array(d["dist_coeffs"]),
            reproj_error=d["reproj_error"],
            image_size=tuple(d["image_size"]),
        )


@dataclass
class StereoParams:
    left: CameraParams = field(default_factory=CameraParams)
    right: CameraParams = field(default_factory=CameraParams)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    T: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    E: np.ndarray = field(default_factory=lambda: np.eye(3))
    F: np.ndarray = field(default_factory=lambda: np.eye(3))
    reproj_error: float = 0.0
    # Wyniki cv2.stereoRectify - potrzebne do szybkiego prostowania w czasie rzeczywistym
    R1: np.ndarray = field(default_factory=lambda: np.eye(3))
    R2: np.ndarray = field(default_factory=lambda: np.eye(3))
    P1: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))
    P2: np.ndarray = field(default_factory=lambda: np.zeros((3, 4)))
    Q: np.ndarray = field(default_factory=lambda: np.eye(4))  # disparity -> depth

    def rectify_maps(
        self, image_size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Zwraca mapy rektyfikacji (map1L, map2L, map1R, map2R).

        Uzycie w petli przechwytywania (raz zainicjalizowane, potem szybkie):
            map1L, map2L, map1R, map2R = stereo.rectify_maps()
            left_rect  = cv2.remap(left_frame,  map1L, map2L, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_frame, map1R, map2R, cv2.INTER_LINEAR)
        """
        size = image_size or self.left.image_size
        map1L, map2L = cv2.initUndistortRectifyMap(
            self.left.camera_matrix, self.left.dist_coeffs,
            self.R1, self.P1, size, cv2.CV_16SC2,
        )
        map1R, map2R = cv2.initUndistortRectifyMap(
            self.right.camera_matrix, self.right.dist_coeffs,
            self.R2, self.P2, size, cv2.CV_16SC2,
        )
        return map1L, map2L, map1R, map2R

    def to_dict(self):
        return {
            "left": self.left.to_dict(), "right": self.right.to_dict(),
            "R": self.R.tolist(), "T": self.T.tolist(),
            "E": self.E.tolist(), "F": self.F.tolist(),
            "reproj_error": self.reproj_error,
            "R1": self.R1.tolist(), "R2": self.R2.tolist(),
            "P1": self.P1.tolist(), "P2": self.P2.tolist(),
            "Q":  self.Q.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            left=CameraParams.from_dict(d["left"]),
            right=CameraParams.from_dict(d["right"]),
            R=np.array(d["R"]), T=np.array(d["T"]),
            E=np.array(d["E"]), F=np.array(d["F"]),
            reproj_error=d["reproj_error"],
            # Klucze opcjonalne - backward compat ze starymi plikami JSON
            R1=np.array(d.get("R1", np.eye(3))),
            R2=np.array(d.get("R2", np.eye(3))),
            P1=np.array(d.get("P1", np.zeros((3, 4)))),
            P2=np.array(d.get("P2", np.zeros((3, 4)))),
            Q=np.array(d.get("Q",  np.eye(4))),
        )


# ---------------------------------------------------------------------------
# Wykrywanie naroznikow
# ---------------------------------------------------------------------------

def _board_points() -> np.ndarray:
    pts = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    pts[:, :2] = np.mgrid[0:BOARD_ROWS, 0:BOARD_COLS].T.reshape(-1, 2)
    return pts * SQUARE_SIZE


def find_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Wykrywa narozniki szachownicy na obrazie.

    Probuje findChessboardCornersSB (subpixel wbudowany, lepszy dla wysokich
    rozdzielczosci), nastepnie wraca do findChessboardCorners + cornerSubPix.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Metoda SB: dokladniejsza, subpixel w jednym przejsciu, brak oddzielnego cornerSubPix
    found, corners = cv2.findChessboardCornersSB(
        gray, (BOARD_ROWS, BOARD_COLS),
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if found:
        return corners

    # Fallback: klasyczna metoda + cornerSubPix
    found, corners = cv2.findChessboardCorners(
        gray, (BOARD_ROWS, BOARD_COLS),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None
    return cv2.cornerSubPix(
        gray, corners,
        config.CORNER_SUBPIX_WIN, config.CORNER_SUBPIX_ZERO_ZONE, CRITERIA,
    )


# ---------------------------------------------------------------------------
# Zbieranie punktow - kazdy obraz przetwarzany dokladnie raz
# ---------------------------------------------------------------------------

def collect_points(image_paths: list[str]) -> CalibrationData:
    """Wczytuje obrazy i wykrywa narozniki szachownicy.

    Kazdy obraz przetwarzany jest dokladnie raz - wyniki przekaz bezposrednio
    do _calibrate_from_data(), aby uniknac ponownego wykrywania.
    """
    obj_points, img_points, img_size = [], [], None
    objp = _board_points()
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            log.warning("Nie mozna wczytac: %s", path)
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        corners = find_corners(img)
        if corners is None:
            log.warning("Brak naroznikow: %s", path)
            continue
        obj_points.append(objp)
        img_points.append(corners)
        log.info("OK: %s", path)
    log.info("Znaleziono wzorzec w %d/%d obrazach", len(img_points), len(image_paths))
    if not img_points:
        raise ValueError("Nie wykryto wzorca na zadnym obrazie")
    assert img_size is not None  # ustawiane przy pierwszym poprawnym obrazie
    return CalibrationData(obj_points, img_points, img_size)


def collect_stereo_points(
    left_paths: list[str], right_paths: list[str]
) -> StereoCalibrationData:
    """Wykrywa narozniki szachownicy w parach stereo.

    Kazdy obraz przetwarzany jest dokladnie raz. Zwraca tylko pary, gdzie
    obie kamery wykryly wzorzec - gotowe do calibrate_stereo / stereoCalibrate.
    """
    objp = _board_points()
    obj_pts, left_pts, right_pts, img_size = [], [], [], None
    for lp, rp in zip(left_paths, right_paths):
        l_img, r_img = cv2.imread(lp), cv2.imread(rp)
        if l_img is None or r_img is None:
            log.warning("Nie mozna wczytac pary: %s, %s", lp, rp)
            continue
        if img_size is None:
            img_size = (l_img.shape[1], l_img.shape[0])
        lc, rc = find_corners(l_img), find_corners(r_img)
        if lc is None or rc is None:
            log.warning("Brak naroznikow w parze: %s, %s", lp, rp)
            continue
        obj_pts.append(objp)
        left_pts.append(lc)
        right_pts.append(rc)
        log.info("OK para: %s | %s", lp, rp)
    log.info("Zgodnych par: %d/%d", len(obj_pts), len(left_paths))
    if not obj_pts:
        raise ValueError("Nie wykryto wzorca na zadnej parze obrazow")
    assert img_size is not None  # ustawiane przy pierwszej poprawnej parze
    return StereoCalibrationData(obj_pts, left_pts, right_pts, img_size)


# ---------------------------------------------------------------------------
# Kalibracja
# ---------------------------------------------------------------------------

def _calibrate_from_data(data: CalibrationData) -> CameraParams:
    """Kalibruje kamere na podstawie juz zebranych punktow (bez ponownego I/O)."""
    if len(data) < config.MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Za malo obrazow z wykrytym wzorcem ({len(data)}), "
            f"min. {config.MIN_CALIBRATION_IMAGES}"
        )
    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        data.obj_points, data.img_points, data.image_size, np.eye(3), np.zeros(5)
    )
    log.info("RMS reproj. error: %.4f px", rms)
    return CameraParams(
        camera_matrix=mtx, dist_coeffs=dist, reproj_error=rms, image_size=data.image_size
    )


def calibrate_single(image_paths: list[str]) -> CameraParams:
    """Kalibruje pojedyncza kamere ze sciezek do obrazow."""
    return _calibrate_from_data(collect_points(image_paths))


def calibrate_stereo(
    left_paths: list[str], right_paths: list[str]
) -> StereoParams:
    """Kalibruje pare stereo.

    Narozniki wykrywane sa dokladnie raz na obraz. Te same punkty trafiaja
    do kalibracji indywidualnych i do stereoCalibrate - brak powtornego I/O.
    Na koncu wywoluje stereoRectify i zapisuje R1/R2/P1/P2/Q w StereoParams,
    dzieki czemu rectify_maps() jest gotowe do uzycia od razu.
    """
    if len(left_paths) != len(right_paths):
        raise ValueError("Liczba obrazow lewej i prawej kamery musi byc rowna")

    stereo_data = collect_stereo_points(left_paths, right_paths)
    if len(stereo_data) < config.MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Za malo par z wzorcem ({len(stereo_data)}), "
            f"min. {config.MIN_CALIBRATION_IMAGES}"
        )

    log.info("Kalibracja stereo na %d parach...", len(stereo_data))
    # Te same punkty co stereo - bez ponownego wykrywania naroznikow
    left_cam = _calibrate_from_data(stereo_data.left)
    right_cam = _calibrate_from_data(stereo_data.right)

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        stereo_data.obj_points,
        stereo_data.left_points,
        stereo_data.right_points,
        left_cam.camera_matrix, left_cam.dist_coeffs,
        right_cam.camera_matrix, right_cam.dist_coeffs,
        stereo_data.image_size, criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    log.info("Stereo RMS reproj. error: %.4f px", rms)

    # stereoRectify: oblicza macierze do rektyfikacji obrazow w czasie rzeczywistym
    # alpha=0: wszystkie piksele po rektyfikacji sa wazne (brak czarnych pasow)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_cam.camera_matrix, left_cam.dist_coeffs,
        right_cam.camera_matrix, right_cam.dist_coeffs,
        stereo_data.image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    return StereoParams(
        left=left_cam, right=right_cam,
        R=R, T=T, E=E, F=F, reproj_error=rms,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    )


# ---------------------------------------------------------------------------
# Zapis / odczyt
# ---------------------------------------------------------------------------

def save_params(params: CameraParams | StereoParams, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params.to_dict(), f, indent=2)
    log.info("Zapisano: %s", path)


@overload
def load_params(path: str, stereo: Literal[False] = False) -> CameraParams: ...


@overload
def load_params(path: str, stereo: Literal[True]) -> StereoParams: ...


def load_params(path: str, stereo: bool = False) -> CameraParams | StereoParams:
    with open(path) as f:
        d = json.load(f)
    return StereoParams.from_dict(d) if stereo else CameraParams.from_dict(d)


def get_image_paths(directory: str, pattern: str | None = None) -> list[str]:
    exts = [pattern] if pattern else config.IMAGE_EXTENSIONS
    for ext in exts:
        paths = sorted(glob.glob(str(Path(directory) / ext)))
        if paths:
            return paths
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kalibracja kamery/stereo")
    parser.add_argument("--mode", choices=["single", "stereo"], default="single")
    parser.add_argument("--left-dir", default=os.path.join(CALIB_DIR, "left"))
    parser.add_argument("--right-dir", default=os.path.join(CALIB_DIR, "right"))
    parser.add_argument("--output", default=os.path.join(CALIB_OUT, "calibration.json"))
    args = parser.parse_args()

    if args.mode == "single":
        imgs = get_image_paths(args.left_dir) or get_image_paths(CALIB_DIR)
        params = calibrate_single(imgs)
    else:
        params = calibrate_stereo(
            get_image_paths(args.left_dir),
            get_image_paths(args.right_dir),
        )
    save_params(params, args.output)
