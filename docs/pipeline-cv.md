# Pipeline wizji komputerowej

Pełny pipeline składa się z 7 etapów — od surowych zdjęć stereo do wymiarów obiektu w milimetrach.

```
Lewy obraz + Prawy obraz + stereo.json
        │
        ▼
  [1] Kalibracja (jednorazowo przed pomiarem)
        │
        ▼
  [2] Rektyfikacja stereo
        │
        ▼
  [3] Mapa dysparycji (SGBM)
        │
        ▼
  [4] Chmura punktów (XYZ [mm])
        │
        ▼
  [5] Wykrywanie płaszczyzny palety (RANSAC)
        │
        ▼
  [6] Segmentacja obiektu
        │
        ▼
  [7] Bounding box + walidacja → W/L/H [mm]
```

---

## Etap 1 — Kalibracja stereo

**Plik:** [calibration.py](../calibration.py)
**Dokumentacja:** [calibration.md](calibration.md)

Jednorazowa operacja wykonywana przed sesją pomiarową. Wyznacza parametry geometryczne pary kamer.
Wyniki zapisywane do `stereo.json` i wczytywane przez wszystkie kolejne etapy.

---

## Etap 2 — Rektyfikacja stereo

**Plik:** [disparity.py](../disparity.py), funkcja `rectify_pair()`

Rektyfikacja transformuje parę zdjęć tak, żeby odpowiadające sobie punkty leżały na tej samej poziomej linii (linia epipolarna). Warunek konieczny dla poprawnego działania SGBM.

```
Obraz lewy (zniekształcony)          Obraz prawy (zniekształcony)
        │                                     │
        └──── cv2.remap(map1L, map2L) ────────┘
                     │         │
             left_rect        right_rect   ← obrazy wyrektyfikowane
```

Mapy rektyfikacji (`map1L, map2L, map1R, map2R`) wyliczane są ze `StereoParams.rectify_maps()`:
- `R1, R2` — macierze obrotu doprowadzające każdą kamerę do wspólnej płaszczyzny
- `P1, P2` — macierze projekcji po rektyfikacji
- Łączy korekcję dystorsji i rektyfikację w jednej operacji `cv2.remap()`

---

## Etap 3 — Mapa dysparycji (SGBM)

**Plik:** [disparity.py](../disparity.py), funkcja `compute_disparity()`

Semi-Global Block Matching (SGBM) szuka dla każdego piksela lewego obrazu odpowiadającego piksela na prawym obrazie. Różnica pozycji (dysparycja) jest odwrotnie proporcjonalna do głębokości.

```
d = xl - xr          (dysparycja w pikselach)
Z = f * B / d        (głębokość w mm)

gdzie:
  f — ogniskowa [px]
  B — baza stereo (odległość między kamerami) [mm]
```

### Parametry SGBM (SGBMConfig)

| Parametr | Wartość domyślna | Opis |
|----------|-----------------|------|
| `min_disparity` | 0 | Minimalna dysparycja [px] |
| `num_disparities` | 64 | Zakres dysparycji (wielokrotność 16) |
| `block_size` | 7 | Rozmiar bloku dopasowania [px] |
| `p1` | 8×3×7² = 1176 | Kara za małe nieciągłości głębokości |
| `p2` | 4×p1 | Kara za duże nieciągłości głębokości |
| `disp12_max_diff` | 1 | Sprawdzenie lewiczo-prawo [px] |
| `uniqueness_ratio` | 10 | Procent unikalności dopasowania |
| `speckle_window_size` | 100 | Okno filtra speckle |
| `speckle_range` | 32 | Zasięg filtra speckle |

**Konwersja dysparycji do głębokości:** `disparity_to_depth(disp, Q)` — mnożenie przez macierz Q metodą `cv2.reprojectImageTo3D`.

---

## Etap 4 — Chmura punktów

**Plik:** [pointcloud.py](../pointcloud.py), funkcja `build_pointcloud()`

Każdy piksel mapy dysparycji jest reprojekcją do przestrzeni 3D przy użyciu macierzy Q:

```
[X, Y, Z, W] = Q * [u, v, d, 1]^T
xyz_mm = [X/W, Y/W, Z/W]
```

**Filtrowanie wejściowe:**
- Odrzucane piksele z `disp == 0` (brak dopasowania)
- Odrzucane piksele z głębokością poza zakresem [50 mm, 5000 mm]
- Odrzucane wartości nieskończone

**Filtrowanie outlierów** — `filter_pointcloud()`:
- Statistical Outlier Removal (k-NN, domyślnie k=20, std_ratio=2.0)
- Usuwa punkty, których średnia odległość do sąsiadów jest > mean + 2σ

**Format wyjściowy:** `(N, 3) float32` w milimetrach + opcjonalnie `(N, 3) uint8` kolory RGB.

---

## Etap 5 — Wykrywanie płaszczyzny palety (RANSAC)

**Plik:** [pallet.py](../pallet.py), funkcja `detect_pallet()`

Paleta EUR/EPAL to dominująca pozioma płaszczyzna w scenie. RANSAC wyznacza ją bez wpływu outlierów (obiektów na palecie).

### Algorytm RANSAC

```
Powtarzaj 1000 razy:
  1. Losuj 3 punkty z chmury
  2. Wyznacz płaszczyznę: ax + by + cz + d = 0
  3. Policz inlierów: punkty z odległością od płaszczyzny < 10 mm
  4. Zapamiętaj najlepszą płaszczyznę (max inlierów)

Refinement SVD:
  Dla najlepszej płaszczyzny wywołaj SVD na inlierach
  → dokładniejszy wektor normalny i centroid
```

### Transformacja do układu palety

`transform_to_pallet_frame()`:
- Obraca chmurę tak, żeby normalny palety pokrywał się z osią Z
- Po transformacji: Z=0 = powierzchnia palety, Z > 0 = obiekty powyżej

### Filtrowanie ROI

`filter_roi()`: Odrzuca punkty spoza obszaru 1200×800 mm (wymiary europalety).

---

## Etap 6 — Segmentacja obiektu

**Plik:** [measurement.py](../measurement.py), funkcja `segment_object()`

Po transformacji do układu palety wystarczy filtr wysokości:

```
obiekt = punkty, gdzie Z > noise_floor_mm   (domyślnie 20 mm)
```

Punkty poniżej `noise_floor` to szum i nierówności powierzchni palety.

---

## Etap 7 — Bounding box + walidacja

**Plik:** [measurement.py](../measurement.py), funkcje `compute_bounding_box()`, `validate_measurement()`

### Bounding Box

```python
width  = max(X) - min(X)   # szerokość [mm]
length = max(Y) - min(Y)   # długość [mm]
height = max(Z) - min(Z)   # wysokość [mm]
```

Opcjonalnie: `extract_3d_contour()` wylicza otoczkę wypukłą (convex hull) rzutu XY.

### Walidacja

Pomiar jest uznawany za poprawny (`validation_passed=True`) jeśli wszystkie kryteria są spełnione:

| Kryterium | Próg | Znaczenie |
|-----------|------|-----------|
| `pallet_rms_mm` | < 50 mm | Płaszczyzna palety wykryta poprawnie |
| `n_pallet_inliers` | ≥ 50 | Wystarczająca liczba punktów palety |
| `pallet_coverage_ratio` | ≥ 10% | Paleta zajmuje przynajmniej 10% chmury |
| `object_height_ok` | 0–1000 mm | Wysokość obiektu w sensownym zakresie |
| `object_within_roi` | — | Obiekt mieści się w obrysie palety |

Niespełnione kryteria trafiają do listy `issues` w odpowiedzi JSON.

---

## Tryb syntetyczny

**Plik:** [pipeline.py](../pipeline.py)

Do testowania bez kamer — `generate_synthetic_scene()` renderuje lewą i prawą kamerę wirtualnej sceny z 3 pudełkami na płaskim tle. Używa idealnych parametrów stereo (f=800 px, baza=120 mm).

```bash
python pipeline.py              # tryb syntetyczny (domyślnie)
python pipeline.py --mode real \
  --calib params.json \
  --left left.png \
  --right right.png
```

Endpoint API: `POST /measure/synthetic` (bez żadnych plików wejściowych).
