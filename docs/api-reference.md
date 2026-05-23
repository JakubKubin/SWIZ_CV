# API Reference

Serwer startuje na `http://localhost:8000`. Interaktywna dokumentacja: `/docs` (Swagger UI).

---

## Sesje

### `POST /sessions`
Tworzy nową sesję pomiarową.

**Response 201:**
```json
{
  "session_id": "a3f9b12c",
  "state": "IDLE",
  "devices": [],
  "created_at": 1715430000.123
}
```

---

### `GET /sessions`
Lista wszystkich aktywnych sesji (do debugowania).

**Response 200:** `list[SessionOut]`

---

### `GET /sessions/{session_id}`
Aktualny stan sesji — urządzenia, stan, liczniki klatek.

**Response 200:** `SessionOut`

**Błąd:** `404` jeśli sesja nie istnieje.

---

### `POST /sessions/{session_id}/join`
Rejestruje urządzenie w sesji.

**Body:**
```json
{
  "device_id": "uuid-leader",
  "mac": "AA:BB:CC:DD:EE:FF",
  "is_leader": true
}
```

**Zasady:**
- Maksymalnie 2 urządzenia na sesję
- Tylko jeden leader (`is_leader: true`)
- Urządzenie może dołączyć do sesji w stanie `IDLE`

**Response 200:** `SessionOut`

**Błędy:** `404` sesja, `409` konflikt (duplikat, sesja pełna, leader już istnieje)

---

### `DELETE /sessions/{session_id}`
Trwale usuwa sesję i wszystkie dane na dysku (kalibracja, zdjęcia, wyniki). Jedyny sposób usunięcia danych sesji.

**Response:** `204 No Content`

---

### `DELETE /sessions/{session_id}/devices/{device_id}`
Wypisuje urządzenie z sesji. Sesja oraz jej dane **nie są usuwane** — pozostają zapisane na dysku (`session.json`), więc urządzenie może później wrócić do sesji i ponownie dołączyć. Trwałe usunięcie: `DELETE /sessions/{id}`.

**Response:** `204 No Content`

---

## Kalibracja

### `POST /sessions/{session_id}/calibration/images`
Przesyła jeden obraz kalibracyjny (szachownica) dla danego urządzenia.

**Body:** `multipart/form-data`
- `device_id` (string) — ID urządzenia
- `file` (binary) — obraz JPG lub PNG

Obrazy są numerowane automatycznie: `frame_0000.jpg`, `frame_0001.jpg`, ...

**Wymagania:** Wywołaj co najmniej 3 razy dla każdego urządzenia.

**Response 201:**
```json
{
  "device_id": "uuid-leader",
  "frame_index": 2,
  "total_frames": 3
}
```

---

### `POST /sessions/{session_id}/calibration/compute`
Uruchamia kalibrację stereo metodą Zhanga w tle.

**Wymagania:**
- 2 urządzenia w sesji
- Co najmniej 3 sparowane klatki kalibracyjne na każde urządzenie

**Response 202:**
```json
{
  "message": "Kalibracja uruchomiona",
  "state": "CALIBRATING"
}
```

Wynik dostępny przez WebSocket event `calibration_done` lub przez `GET /calibration`.

**Błędy:** `409` za mało urządzeń / klatek / kalibracja już trwa

---

### `GET /sessions/{session_id}/calibration`
Status kalibracji.

**Response 200:**
```json
{
  "state": "READY",
  "reproj_error": 1.247,
  "message": "Kalibracja OK (reproj_error=1.247 px)"
}
```

`reproj_error` to błąd reprojekcji w pikselach — im niższy, tym lepsza kalibracja.
Wartość > 2.0 px sugeruje ponowną kalibrację.

---

## Przechwytywanie

### `POST /sessions/{session_id}/capture/trigger`
Rozsyła do wszystkich urządzeń komendę jednoczesnego przechwycenia.

**Wymagania:** Sesja w stanie `READY` lub `DONE`.

**Body:**
```json
{
  "delay_ms": 500
}
```

Serwer wylicza `target_ts = now() + delay_ms/1000` i wysyła przez WebSocket do wszystkich urządzeń.

**Response 200:**
```json
{
  "at": 1715430500.623,
  "delay_ms": 500
}
```

Urządzenia otrzymują WebSocket event:
```json
{"event": "capture_trigger", "at": 1715430500.623, "delay_ms": 500}
```

---

### `POST /sessions/{session_id}/capture/images`
Przesyła zdjęcie pomiarowe z danego urządzenia.

**Wymagania:** Sesja w stanie `READY` lub `DONE`.

**Body:** `multipart/form-data`
- `device_id` (string)
- `file` (binary) — zdjęcie JPG lub PNG

Pliki są numerowane: `capture_0000.jpg`, `capture_0001.jpg`, ...
Pipeline zawsze używa najnowszego zdjęcia.

**Response 201:**
```json
{
  "device_id": "uuid-follower",
  "frame_index": 0,
  "total_frames": 1
}
```

---

## Pomiar

### `POST /sessions/{session_id}/measure`
Uruchamia pełny pipeline pomiaru 3D w tle.

**Wymagania:**
- Sesja w stanie `READY` (po kalibracji)
- Co najmniej 1 zdjęcie pomiarowe per urządzenie

**Response 202:**
```json
{
  "message": "Pomiar uruchomiony",
  "state": "PROCESSING"
}
```

Wynik dostępny przez WebSocket event `measurement_done` lub przez `GET /measurement`.

---

### `GET /sessions/{session_id}/measurement`
Wyniki ostatniego pomiaru.

**Response 200:**
```json
{
  "validation_passed": true,
  "width_mm": 450.3,
  "length_mm": 320.7,
  "height_mm": 280.1,
  "pallet_rms_mm": 8.4,
  "n_object_pts": 12400,
  "n_pallet_inliers": 3200,
  "issues": [],
  "report": "=== Raport pomiaru ===\n..."
}
```

**Błąd:** `404` jeśli pomiar jeszcze nie był uruchamiany.

---

### `GET /sessions/{session_id}/measurement/report`
Pełny raport tekstowy jako `text/plain`.

---

## Narzędzia

### `POST /measure/synthetic`
Uruchamia pełny pipeline na danych syntetycznych (bez kamer, do testów API).

Generuje wirtualną scenę 3 pudełek na tle, przetwarza przez SGBM.

**Response 200:** `MeasurementOut`
**Błąd:** `422` jeśli detekcja palety nie powiodła się na scenie syntetycznej.

---

### `GET /health`
Health check.

**Response 200:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

## WebSocket: `WS /ws/{session_id}/{device_id}`

Otworzyć po zarejestrowaniu urządzenia przez `POST /join`. Jeden WebSocket per urządzenie.

### Zamknięcie połączenia

| Kod | Przyczyna |
|-----|-----------|
| `4003` | `device_id` nie jest w sesji |
| `4004` | Sesja `session_id` nie istnieje |

### Klient → Serwer

```json
{"action": "ping"}
```
Odpowiedź: `{"event": "pong", "t": 1715430500.0}`

```json
{"action": "captured", "at": 1715430500.623}
```
Broadcast do wszystkich: `{"event": "device_captured", "device_id": "...", "at": ...}`

### Serwer → Klienci (broadcast)

| Event | Kiedy | Pola |
|-------|-------|------|
| `device_joined` | urządzenie dołączyło | `device_id`, `is_leader`, `total_devices` |
| `device_left` | urządzenie się rozłączyło | `device_id`, `remaining` |
| `capture_trigger` | trigger przechwycenia | `at` (timestamp), `delay_ms` |
| `calibration_done` | kalibracja zakończona | `reproj_error` |
| `measurement_done` | pomiar zakończony | `width_mm`, `length_mm`, `height_mm`, `validation_passed` |
| `error` | błąd kalibracji lub pomiaru | `message` |
