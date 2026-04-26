# Sesja testowa: c6383b0e

Pierwsza rzeczywista sesja pomiarowa systemu SWIZ – Zespół 3.
Wykonana w ramach etapu M2 projektu. Celem sesji była weryfikacja
poprawności działania kalibracji, pipeline'u obliczeniowego oraz
integracji backendu z aplikacją Flutter.

## Co zostało zweryfikowane i działa

| Moduł                           | Wynik                                      |
|---------------------------------|--------------------------------------------|
| Kalibracja stereo (metoda Zhanga) | RMS = 1.7247 px – wynik poprawny         |
| Rektyfikacja par stereo         | `epipolar_check.png` – linie wyrównane     |
| Mapa dysparycji (SGBM)          | Obliczona poprawnie, zakres 0.6–63.0 px    |
| Backend FastAPI + WebSocket     | Działa pod http://localhost:8000/docs      |
| Aplikacja Flutter (web)         | Działa pod http://192.168.0.251:3000       |
| Pliki wynikowe pipeline         | PNG, PLY, report.txt zapisane poprawnie    |

Kalibracja indywidualna obu kamer osiągnęła bardzo niski błąd reprojekcji
(lewa: 0.2810 px, prawa: 0.2702 px), co potwierdza poprawność implementacji
modułu kalibracji i wykrywania narożników szachownicy.

## Parametry kalibracji

| Parametr                        | Wartość          |
|---------------------------------|------------------|
| Błąd reprojekcji – lewa kamera  | 0.2810 px        |
| Błąd reprojekcji – prawa kamera | 0.2702 px        |
| Błąd reprojekcji stereo (RMS)   | 1.7247 px        |
| Baza stereo \|T\|               | 165.2 mm         |
| Ogniskowa fx (lewa)             | 1918.4 px        |
| Ogniskowa fy (lewa)             | 1920.7 px        |
| Rozdzielczość zdjęć             | 2560 × 1920 px   |
| Liczba par kalibracyjnych       | 10               |

## Wyniki pipeline

| Etap                        | Status | Uwagi                                         |
|-----------------------------|--------|-----------------------------------------------|
| Kalibracja stereo           | ✓ OK   | RMS = 1.7247 px                               |
| Rektyfikacja                | ✓ OK   | epipolar_check.png poprawny                   |
| Mapa dysparycji (SGBM)      | ✓ OK   | 24% ważnych pikseli (660 902 / 2 764 800)     |
| Mapa głębokości             | ~ częściowo | Wymagała ręcznej korekty macierzy Q     |
| Chmura punktów              | ~ częściowo | Problemy z ustawieniem telefonów (patrz niżej)|
| Detekcja palety (RANSAC)    | – niedostępna | Wymaga poprawnej chmury punktów         |
| Pomiar wymiarów             | – niedostępna | Wymaga detekcji palety                  |

Pliki wynikowe: `pipeline_output/`

## Napotkane trudności

### Ustawienie telefonów – główne wyzwanie sesji
Uzyskany wektor translacji między kamerami wyniósł `T = [68, 21, 148] mm`.
Duża składowa Z (148 mm) wskazuje, że jeden telefon był wysunięty do przodu
względem drugiego. W prawidłowej konfiguracji translacja powinna mieć postać
`T = [~150, 0, 0] mm` – wyłącznie składowa X (przesunięcie poziome).
Efektem są linie epipolarne nierównoległe do poziomu, co utrudnia działanie SGBM.

### Spójność rozdzielczości
Kalibracja była wykonana w rozdzielczości 2560 px, natomiast zdjęcia pomiarowe
miały inną rozdzielczość. Powoduje to błędne wartości w macierzy Q.
Macierz Q w `stereo.json` została ręcznie skorygowana po sesji.

### Pokrycie mapy dysparycji – 24% ważnych pikseli
SGBM do znalezienia dopasowań między obrazami potrzebuje tekstury na powierzchni
mierzonych obiektów. Użyte kartony były zbyt gładkie i jednolite – algorytm nie
mógł znaleźć wystarczającej liczby charakterystycznych punktów. Docelowo należy
używać kartonów z nadrukami lub nakleić wzory.

## Instrukcja prawidłowego wykonania zdjęć

### Szachownica kalibracyjna
- Wymiary: **9×6 kwadratów** (= 8×5 wewnętrznych narożników – tyle podajemy w konfiguracji)
- Rozmiar kwadratu: **15 mm**
- Kolor: czarno-biała, **bez kolorowych ramek** – OpenCV wymaga białego tła dookoła
- Przyklejona na sztywną tekturę – nie może się wyginać
- Wygenerować na: https://calib.io (Rows: 9, Cols: 6, Checker Width: 15 mm)

### Ustawienie telefonów
- Oba telefony przyklejone taśmą do **sztywnego kartonu w jednej poziomej linii**
- Odstęp między telefonami: **~15 cm**
- Ten sam poziom – żadnych przesunięć w górę/dół ani do przodu/tyłu
- **Nie ruszać telefonów przez całą sesję** – ani podczas kalibracji, ani podczas pomiaru
- Po kalibracji sprawdzić wektor T: powinien mieć postać `[~150, 0, 0]` mm

### Zdjęcia kalibracyjne
- Telefony pionowo (portrait)
- **Minimum 15 par zdjęć**, docelowo 20–30
- Tylko szachownica się rusza, nie telefony
- Jedna osoba trzyma szachownicę i zmienia pozycję
- Druga osoba naciska wyzwalacz jednocześnie na obu telefonach
- Pozycje szachownicy: środek kadru, lewy róg, prawy róg, przechylona we wszystkich
  kierunkach, różne odległości od kamer

### Zdjęcia pomiarowe
- Telefony w **tej samej pozycji** co podczas kalibracji – nie ruszać!
- Rozdzielczość musi być identyczna jak przy kalibracji
- Pudełka poustawiane jak ładunek na palecie, różne wysokości
- **Kartony z nadrukami/wzorami** – nie gładkie (SGBM potrzebuje tekstury)
- Odległość obiektów od telefonów: **~60 cm**
- Dobre oświetlenie od przodu, bez silnych cieni
- Obie osoby wyzwalają jednocześnie

## Priorytety na kolejną sesję

1. **Ustawienie telefonów** – przykleić do sztywnego podłoża, zweryfikować T po kalibracji
2. **Spójność rozdzielczości** – kalibracja i pomiar w tej samej rozdzielczości
3. **Więcej par kalibracyjnych** – docelowo 20–30 par zamiast 10
4. **Tekstura pudełek** – przykleić gazety lub wydruki na kartony

