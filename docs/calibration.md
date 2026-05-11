# Kalibracja stereo

Kalibracja wyznacza geometrię układu dwóch kamer — parametry wewnętrzne każdej kamery oraz ich wzajemne położenie w przestrzeni. Bez kalibracji nie można przeliczać dysparycji na milimetry.

---

## Metoda Zhanga

Metoda Zhanga (2000) to standardowa technika kalibracji kamer ze wzorcem planarnymi (szachownica).

**Algorytm:**
1. Fotografuj szachownicę z różnych kątów i odległości (min. 3 pozycje, rekomendowane 10–20)
2. Wykryj narożniki szachownicy na każdym zdjęciu z precyzją subpikselową
3. Każde zdjęcie daje równania wiążące znane współrzędne 3D narożników z ich pozycją 2D na obrazie
4. Minimalizuj błąd reprojekcji (RMS) optymalizując macierz K i współczynniki dystorsji

---

## Parametry szachownicy (.env)

Skonfiguruj przed użyciem w pliku `.env`:

```env
CHECKERBOARD_ROWS=9      # liczba wewnętrznych narożników — wiersze
CHECKERBOARD_COLS=6      # liczba wewnętrznych narożników — kolumny
SQUARE_SIZE_MM=25.0      # fizyczny rozmiar kwadratu [mm]
```

**Ważne:** `ROWS` i `COLS` to liczba **wewnętrznych narożników**, nie kwadratów.
Szachownica 10×7 kwadratów ma 9×6 wewnętrznych narożników.

---

## Krok po kroku — kalibracja przez API

### 1. Utwórz sesję i zarejestruj urządzenia

```bash
# Utwórz sesję
curl -X POST http://localhost:8000/sessions
# → {"session_id": "a3f9b12c", ...}

# Zarejestruj lewe urządzenie (lider)
curl -X POST http://localhost:8000/sessions/a3f9b12c/join \
  -H "Content-Type: application/json" \
  -d '{"device_id": "dev-left", "mac": "AA:BB:CC:DD:EE:FF", "is_leader": true}'

# Zarejestruj prawe urządzenie
curl -X POST http://localhost:8000/sessions/a3f9b12c/join \
  -H "Content-Type: application/json" \
  -d '{"device_id": "dev-right", "mac": "11:22:33:44:55:66", "is_leader": false}'
```

### 2. Prześlij obrazy kalibracyjne

Dla każdego urządzenia prześlij co najmniej 3 zdjęcia szachownicy (rekomendowane 10–20):

```bash
# Zdjęcie kalibracyjne z lewej kamery
curl -X POST http://localhost:8000/sessions/a3f9b12c/calibration/images \
  -F "device_id=dev-left" \
  -F "file=@left_calib_01.jpg"

# Powtórz dla każdego kolejnego zdjęcia i dla dev-right
```

Ważne zasady fotografowania szachownicy:
- Różne kąty nachylenia (30°–60° od osi kamery)
- Różne odległości
- Szachownica wypełniająca przynajmniej połowę kadru
- Obie kamery widoczne na tym samym zdjęciu (pary synchroniczne)

### 3. Uruchom kalibrację

```bash
curl -X POST http://localhost:8000/sessions/a3f9b12c/calibration/compute
# → {"message": "Kalibracja uruchomiona", "state": "CALIBRATING"}
```

### 4. Sprawdź wynik

Przez polling lub WebSocket event `calibration_done`:

```bash
curl http://localhost:8000/sessions/a3f9b12c/calibration
# → {"state": "READY", "reproj_error": 1.24, "message": "Kalibracja OK"}
```

---

## Kalibracja z wiersza poleceń (lokalna)

```bash
# Kalibracja stereo z katalogu z parami zdjęć
python calibration.py --mode stereo \
  --left-dir calib_images/left \
  --right-dir calib_images/right \
  --output calib_output/stereo.json

# Kalibracja pojedynczej kamery
python calibration.py --mode single \
  --left-dir calib_images/left \
  --output calib_output/left.json
```

---

## Format pliku stereo.json

```json
{
  "left": {
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_coeffs": [k1, k2, p1, p2, k3],
    "reproj_error": 0.847,
    "image_size": [1920, 1080]
  },
  "right": { ... },
  "R": [[...], [...], [...]],   // rotacja lewej względem prawej kamery
  "T": [[tx], [ty], [tz]],      // translacja (baza stereo w osi X ≈ -120 mm)
  "E": [[...], ...],            // macierz esencjalna
  "F": [[...], ...],            // macierz fundamentalna
  "reproj_error": 1.24,         // stereo RMS [px]
  "R1": [[...], ...],           // macierz rektyfikacji lewej kamery
  "R2": [[...], ...],           // macierz rektyfikacji prawej kamery
  "P1": [[...], ...],           // macierz projekcji po rektyfikacji (lewa)
  "P2": [[...], ...],           // macierz projekcji po rektyfikacji (prawa)
  "Q":  [[...], ...]            // macierz reprojekcji 4×4 (dysparycja → 3D [mm])
}
```

---

## Interpretacja błędu RMS

Błąd reprojekcji (RMS) mierzy jak dobrze model kamery odwzorowuje obserwacje — ile pikseli różni się projekcja 3D punktu od rzeczywistej pozycji na obrazie.

| Wartość RMS | Interpretacja |
|-------------|--------------|
| < 0.5 px | Doskonała kalibracja |
| 0.5–1.0 px | Dobra (typowe dla telefonów) |
| 1.0–2.0 px | Akceptowalna |
| > 2.0 px | Zła — wymagana ponowna kalibracja |

Progi w systemie ([config.py](../config.py)):
- Pojedyncza kamera: `MAX_SINGLE_REPROJ_ERROR = 1.0 px`
- Stereo: `MAX_STEREO_REPROJ_ERROR = 2.0 px`

### Powody złego RMS

- Zbyt mało zdjęć (minimum 3, rekomendowane 10–20)
- Szachownica niewyraźna (rozmazana, prześwietlona)
- Zdjęcia tylko z jednego kąta
- Szachownica zbyt mała w kadrze
- Ruch kamery lub szachownicy podczas eksponowania

---

## Parametry wewnętrzne kamery (macierz K)

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

fx, fy — ogniskowe [px] — typowo 1000–2000 px dla telefonów
cx, cy — punkt główny — zwykle ≈ (width/2, height/2)
```

Współczynniki dystorsji `[k1, k2, p1, p2, k3]`:
- `k1, k2, k3` — dystorsja promieniowa (beczkowanie / poduszka)
- `p1, p2` — dystorsja tangencjalna (decentracja obiektywu)

---

## Detekcja narożników — strategia dwuetapowa

Funkcja `find_corners()` w [calibration.py](../calibration.py):

1. **findChessboardCornersSB** — nowsza metoda OpenCV z wbudowaną precyzją subpikselową, lepsza dla wysokich rozdzielczości
2. **Fallback:** `findChessboardCorners` + `cornerSubPix` — klasyczna metoda, gdy SB nie znajduje wzorca

Obrazy > 1920 px szerokości są skalowane przed detekcją (wymagania wydajnościowe przy rozdzielczości telefonów 4K+).
