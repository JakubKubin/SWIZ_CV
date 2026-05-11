# Aplikacja Flutter

Aplikacja mobilna do zarządzania sesją, kalibracji, wyzwalania pomiaru i wyświetlania wyników.
Docelowe platformy: Android i iOS. Dostępna też jako web (do testów UI/UX).

---

## Struktura katalogów

```
flutter_app/
├── pubspec.yaml                  # zależności
├── lib/
│   ├── main.dart                 # punkt wejścia, Provider setup
│   ├── models/
│   │   └── models.dart           # klasy danych (DeviceInfo, SessionData, MeasurementResult)
│   ├── providers/
│   │   └── app_state.dart        # globalny stan (Provider pattern)
│   ├── services/
│   │   └── api_service.dart      # klient HTTP do backendu
│   ├── screens/
│   │   ├── home_screen.dart      # ekran startowy
│   │   ├── session_screen.dart   # zarządzanie sesją
│   │   ├── calibration_screen.dart # kalibracja
│   │   ├── capture_screen.dart   # wyzwalanie zdjęcia
│   │   └── results_screen.dart   # wyniki pomiaru
│   ├── theme/
│   │   └── app_theme.dart        # motyw Material Design
│   └── widgets/
│       └── app_banner.dart       # bannery statusu
├── android/                      # konfiguracja Android
├── ios/                          # konfiguracja iOS
└── web/                          # konfiguracja web
```

---

## Zależności (pubspec.yaml)

| Pakiet | Zastosowanie |
|--------|-------------|
| `http` | Wywołania REST API |
| `web_socket_channel` | WebSocket (działa na mobile i web) |
| `image_picker` | Dostęp do kamery (Camera2 API / AVFoundation) |
| `provider` | Zarządzanie stanem (Provider pattern) |
| `shared_preferences` | Zapisywanie URL serwera, device_id, MAC |

---

## Modele danych (lib/models/models.dart)

### DeviceInfo

```dart
class DeviceInfo {
  final String deviceId;
  final String mac;
  final bool isLeader;
  final double joinedAt;       // Unix timestamp
  final bool wsConnected;      // czy WebSocket aktywny
  final int calibFrameCount;   // liczba przesłanych klatek kalibracyjnych
  final int captureFrameCount; // liczba przesłanych zdjęć pomiarowych
}
```

### SessionData

```dart
class SessionData {
  final String sessionId;
  final String state;           // IDLE / CALIBRATING / READY / PROCESSING / DONE
  final List<DeviceInfo> devices;
  final double createdAt;
}
```

Gotowe gettery do sprawdzania stanu:
```dart
session.isIdle          // state == 'IDLE'
session.isCalibrating   // state == 'CALIBRATING'
session.isReady         // state == 'READY'
session.isDone          // state == 'DONE'
session.allCaptured     // wszystkie urządzenia mają ≥1 zdjęcie
session.minCalibFrames  // min. klatek kalibracyjnych spośród urządzeń
```

### MeasurementResult

```dart
class MeasurementResult {
  final bool validationPassed;
  final double widthMm, lengthMm, heightMm;
  final double palletRmsMm;      // residuum płaszczyzny palety
  final int nObjectPts;          // punkty obiektu
  final int nPalletInliers;      // inliery RANSAC palety
  final List<String> issues;     // lista problemów
  final String report;           // raport tekstowy
}
```

### ApiException

```dart
class ApiException implements Exception {
  final int statusCode;  // HTTP status
  final String message;  // treść błędu z backendu
}
```

---

## ApiService (lib/services/api_service.dart)

Wszystkie wywołania do backendu. Instancja tworzona z URL serwera:

```dart
final api = ApiService('http://192.168.1.100:8000');
```

### Mapowanie metod na endpointy

| Metoda Dart | Endpoint | Opis |
|-------------|----------|------|
| `healthCheck()` | `GET /health` | Sprawdza czy serwer działa (timeout 5s) |
| `createSession()` | `POST /sessions` | Tworzy nową sesję |
| `joinSession(sid, deviceId, mac, isLeader)` | `POST /sessions/{sid}/join` | Rejestruje urządzenie |
| `getSession(sid)` | `GET /sessions/{sid}` | Pobiera stan sesji |
| `deleteSession(sid)` | `DELETE /sessions/{sid}` | Usuwa sesję |
| `leaveDevice(sid, deviceId)` | `DELETE /sessions/{sid}/devices/{deviceId}` | Opuszcza sesję |
| `listSessions()` | `GET /sessions` | Lista sesji (debug) |
| `uploadCalibImage(sid, deviceId, bytes)` | `POST /sessions/{sid}/calibration/images` | Prześlij klatki kalibracyjne |
| `computeCalibration(sid)` | `POST /sessions/{sid}/calibration/compute` | Uruchamia kalibrację |
| `getCalibrationStatus(sid)` | `GET /sessions/{sid}/calibration` | Status kalibracji |
| `triggerCapture(sid, delayMs)` | `POST /sessions/{sid}/capture/trigger` | Rozsyła trigger zdjęcia |
| `uploadCaptureImage(sid, deviceId, bytes)` | `POST /sessions/{sid}/capture/images` | Przesyła zdjęcie pomiarowe |
| `runMeasurement(sid)` | `POST /sessions/{sid}/measure` | Uruchamia pipeline 3D |
| `getMeasurement(sid)` | `GET /sessions/{sid}/measurement` | Pobiera wyniki |
| `getMeasurementReport(sid)` | `GET /sessions/{sid}/measurement/report` | Pełny raport tekstowy |
| `syntheticMeasure()` | `POST /measure/synthetic` | Test bez kamer |

Upload plików używa `http.MultipartRequest` — zdjęcia wysyłane jako `multipart/form-data`.

---

## Ekrany

### HomeScreen (home_screen.dart)
- Konfiguracja URL serwera
- Przycisk „Utwórz sesję" (POST /sessions)
- Przycisk „Dołącz do sesji" (wpisanie session_id)
- Status połączenia (GET /health)

### SessionScreen (session_screen.dart)
- Wyświetla listę podłączonych urządzeń
- Status sesji (IDLE / CALIBRATING / READY / PROCESSING / DONE)
- Liczniki klatek kalibracyjnych i pomiarowych per urządzenie
- Wskaźnik połączenia WebSocket

### CalibrationScreen (calibration_screen.dart)
- Przycisk „Zrób zdjęcie szachownicy" (image_picker → kamera)
- Upload zdjęcia przez `uploadCalibImage()`
- Licznik przesłanych klatek
- Przycisk „Oblicz kalibrację" (gdy ≥3 klatki)
- Wyświetla błąd reprojekcji RMS po kalibracji

### CaptureScreen (capture_screen.dart)
- Przycisk „Trigger" (tylko lider): wywołuje `triggerCapture()`
- Oczekiwanie na WebSocket event `capture_trigger`
- Automatyczne robienie zdjęcia o czasie `at` (timestamp)
- Upload przez `uploadCaptureImage()`
- Przycisk „Uruchom pomiar" → `runMeasurement()`

### ResultsScreen (results_screen.dart)
- Wyświetla W/L/H w mm
- Status walidacji (zielony/czerwony)
- Lista problemów (`issues`)
- Pełny raport tekstowy

---

## State management (app_state.dart)

Provider pattern — `AppState` jako `ChangeNotifier`:

```dart
class AppState extends ChangeNotifier {
  String serverUrl;    // URL backendu (zapisywany w SharedPreferences)
  String deviceId;     // UUID urządzenia (generowany przy pierwszym uruchomieniu)
  String mac;          // adres MAC
  SessionData? session; // aktualna sesja
  MeasurementResult? lastResult; // ostatni wynik
}
```

Dostęp w widgetach:
```dart
final state = context.watch<AppState>();
final state = context.read<AppState>();
```

---

## WebSocket

Połączenie przez pakiet `web_socket_channel`:

```dart
final channel = WebSocketChannel.connect(
  Uri.parse('ws://host:8000/ws/$sessionId/$deviceId'),
);

channel.stream.listen((message) {
  final data = jsonDecode(message as String);
  switch (data['event']) {
    case 'capture_trigger':
      // Zaplanuj zdjęcie na data['at'] (Unix timestamp)
      break;
    case 'calibration_done':
      // Odśwież ekran kalibracji
      break;
    case 'measurement_done':
      // Przejdź do ekranu wyników
      break;
  }
});
```

---

## Uruchomienie

```bash
cd flutter_app

# Instalacja zależności
flutter pub get

# Web (do testów)
flutter run -d chrome

# Android (fizyczne urządzenie lub emulator)
flutter run -d android

# iOS (wymaga macOS + Xcode)
flutter run -d ios

# Build release APK
flutter build apk --release
```

**Uwaga:** URL serwera musi być osiągalny z urządzenia. Na fizycznym telefonie użyj IP komputera w sieci lokalnej, np. `http://192.168.1.100:8000`.
