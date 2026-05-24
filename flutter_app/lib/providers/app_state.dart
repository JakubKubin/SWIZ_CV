// app_state.dart - główny ChangeNotifier zarządzający stanem aplikacji
// Obsługuje: sesję, WebSocket, kalibrację, przechwytywanie, pomiar.

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../utils/log.dart';

class AppState extends ChangeNotifier {
  static const _log = Log('AppState');

  static const _kServerUrl = 'server_url';
  static const _kDeviceId = 'device_id';
  static const _kMac = 'mac';
  static const _kSessions = 'known_sessions';

  final SharedPreferences _prefs;

  AppState(this._prefs) {
    _serverUrl = _prefs.getString(_kServerUrl) ?? 'http://192.168.1.1:8000';
    deviceId = _prefs.getString(_kDeviceId) ?? _generateDeviceId();
    mac = _prefs.getString(_kMac) ?? _generateMac();
    _prefs.setString(_kDeviceId, deviceId);
    _prefs.setString(_kMac, mac);
    _loadKnownSessions();
  }

  // -------------------------------------------------------------------------
  // Konfiguracja serwera
  // -------------------------------------------------------------------------

  String _serverUrl = 'http://192.168.1.1:8000';

  String get serverUrl => _serverUrl;

  set serverUrl(String v) {
    _serverUrl = v;
    _prefs.setString(_kServerUrl, v);
  }

  // -------------------------------------------------------------------------
  // Tożsamość urządzenia
  // -------------------------------------------------------------------------

  String deviceId = '';
  String mac = '';
  bool isLeader = true;
  bool isCamera = true;

  // -------------------------------------------------------------------------
  // Stan sesji
  // -------------------------------------------------------------------------

  String? sessionId;
  SessionData? session;
  MeasurementResult? measurement;

  /// Lokalna historia sesji utworzonych/dołączonych z tego urządzenia.
  /// Najnowsze na początku listy.
  List<SessionRef> knownSessions = [];

  /// To urządzenie w bieżącej sesji (lub null, jeśli nie figuruje w niej).
  DeviceInfo? get myDevice {
    final devices = session?.devices;
    if (devices == null) return null;
    for (final d in devices) {
      if (d.deviceId == deviceId) return d;
    }
    return null;
  }

  // -------------------------------------------------------------------------
  // Stan UI
  // -------------------------------------------------------------------------

  bool isLoading = false;
  String? error;
  String? info;

  // -------------------------------------------------------------------------
  // WebSocket
  // -------------------------------------------------------------------------

  WebSocketChannel? _ws;
  StreamSubscription<dynamic>? _wsSub;

  /// Offset: czas serwera − czas lokalny (sekundy). Do synchronizacji przechwycenia.
  double serverTimeOffset = 0.0;

  /// Czas lokalny wysłania ostatniego pinga (Unix sek.) — do korekty RTT.
  double _pingSentAt = 0.0;

  /// true gdy WebSocket jest aktywny i odebrał co najmniej jeden pong.
  bool wsConnected = false;

  /// Timestamp (Unix sek.) kiedy urządzenie powinno zrobić zdjęcie.
  double? captureTriggerAt;

  /// Timestamp (Unix sek.) kiedy urządzenie powinno zrobić zdjęcie kalibracyjne.
  double? calibTriggerAt;

  /// Log ostatnich 30 zdarzeń WebSocket (do widoku debugowania).
  final List<Map<String, dynamic>> wsLog = [];

  // -------------------------------------------------------------------------
  // Pomocnicy
  // -------------------------------------------------------------------------

  ApiService get _api => ApiService(serverUrl);

  void _setLoading(bool v) {
    isLoading = v;
    notifyListeners();
  }

  void _setError(String msg) {
    _log.warn('Błąd UI: $msg');
    error = msg;
    isLoading = false;
    notifyListeners();
  }

  void _setInfo(String msg) {
    _log.info(msg);
    info = msg;
    notifyListeners();
  }

  void clearError() {
    error = null;
    notifyListeners();
  }

  void clearInfo() {
    info = null;
    notifyListeners();
  }

  void clearCaptureTrigger() {
    captureTriggerAt = null;
    // Nie wywołujemy notifyListeners - caller zrobi to sam po starcie countdown.
  }

  void clearCalibTrigger() {
    calibTriggerAt = null;
  }

  // -------------------------------------------------------------------------
  // Połączenie / health
  // -------------------------------------------------------------------------

  Future<bool> testConnection() async {
    return ApiService(serverUrl).healthCheck();
  }

  // -------------------------------------------------------------------------
  // Historia sesji (SharedPreferences)
  // -------------------------------------------------------------------------

  void _loadKnownSessions() {
    final raw = _prefs.getString(_kSessions);
    if (raw == null || raw.isEmpty) return;
    try {
      final list = jsonDecode(raw) as List;
      knownSessions = list
          .map((e) => SessionRef.fromJson(e as Map<String, dynamic>))
          .toList();
      _log.info('Wczytano ${knownSessions.length} zapamiętanych sesji');
    } catch (e, st) {
      _log.warn('Nie udało się odczytać historii sesji - czyszczę', e, st);
      knownSessions = [];
    }
  }

  void _persistKnownSessions() {
    _prefs.setString(
        _kSessions, jsonEncode(knownSessions.map((s) => s.toJson()).toList()));
  }

  /// Zapamiętuje sesję w historii (lub aktualizuje istniejący wpis).
  void _rememberSession(String sid, bool leader, {bool camera = true}) {
    knownSessions.removeWhere((s) => s.sessionId == sid);
    knownSessions.insert(
      0,
      SessionRef(
        sessionId: sid,
        serverUrl: serverUrl,
        isLeader: leader,
        isCamera: camera,
        createdAt: DateTime.now().millisecondsSinceEpoch / 1000.0,
      ),
    );
    _persistKnownSessions();
  }

  void _forgetSession(String sid) {
    knownSessions.removeWhere((s) => s.sessionId == sid);
    _persistKnownSessions();
  }

  /// Czyści dane przejściowe przy przełączaniu między sesjami, żeby nie
  /// pokazać nieaktualnych wyników/zdarzeń z poprzedniej sesji.
  void _resetTransientState() {
    measurement = null;
    captureTriggerAt = null;
    calibTriggerAt = null;
    serverTimeOffset = 0.0;
    wsConnected = false;
    wsLog.clear();
    error = null;
    info = null;
  }

  // -------------------------------------------------------------------------
  // Tworzenie i dołączanie do sesji
  // -------------------------------------------------------------------------

  /// Tworzy nową sesję i dołącza jako lider lub follower.
  Future<bool> createAndJoin(String did, String m, bool leader, {bool camera = true}) async {
    _setLoading(true);
    try {
      serverUrl = serverUrl.trim();
      deviceId = did;
      mac = m;
      isLeader = leader;
      isCamera = camera;
      _resetTransientState();

      final sess = await _api.createSession();
      sessionId = sess.sessionId;

      final joined = await _api.joinSession(sess.sessionId, did, m, leader, isCamera: camera);
      session = joined;
      _rememberSession(sess.sessionId, leader, camera: camera);

      _log.info('Utworzono i dołączono do sesji ${sess.sessionId} '
          '(leader=$leader, device=$did)');
      _connectWs();
      _setLoading(false);
      return true;
    } catch (e, st) {
      _log.warn('Tworzenie/dołączanie do sesji nie powiodło się '
          '(device=$did, server=$serverUrl)', e, st);
      sessionId = null;
      _setError(e.toString());
      return false;
    }
  }

  /// Dołącza do istniejącej sesji (follower).
  Future<bool> joinExisting(
      String sid, String did, String m, bool leader, {bool camera = true}) async {
    _setLoading(true);
    try {
      serverUrl = serverUrl.trim();
      deviceId = did;
      mac = m;
      isLeader = leader;
      isCamera = camera;
      sessionId = sid;
      _resetTransientState();

      final joined = await _api.joinSession(sid, did, m, leader, isCamera: camera);
      session = joined;
      _rememberSession(sid, leader, camera: camera);

      _log.info('Dołączono do istniejącej sesji $sid (leader=$leader, device=$did)');
      _connectWs();
      _setLoading(false);
      return true;
    } catch (e, st) {
      _log.warn('Dołączanie do sesji $sid nie powiodło się (device=$did)', e, st);
      sessionId = null;
      _setError(e.toString());
      return false;
    }
  }

  /// Wraca do zapamiętanej sesji: weryfikuje że istnieje na serwerze,
  /// w razie potrzeby ponownie rejestruje urządzenie, łączy WebSocket
  /// i pobiera wynik pomiaru jeśli sesja jest zakończona.
  Future<bool> resumeSession(SessionRef ref) async {
    _setLoading(true);
    try {
      serverUrl = ref.serverUrl.isNotEmpty ? ref.serverUrl : serverUrl;
      isLeader = ref.isLeader;
      sessionId = ref.sessionId;
      _resetTransientState();

      var fetched = await _api.getSession(ref.sessionId);

      // Jeśli to urządzenie nie figuruje już w sesji (np. wcześniej ją
      // opuściło), dołączamy je ponownie tą samą rolą.
      final stillMember = fetched.devices.any((d) => d.deviceId == deviceId);
      if (!stillMember) {
        _log.info('Urządzenie $deviceId nie jest już w sesji ${ref.sessionId} '
            '- ponowna rejestracja (leader=${ref.isLeader}, camera=${ref.isCamera})');
        fetched = await _api.joinSession(
          ref.sessionId, deviceId, mac, ref.isLeader,
          isCamera: ref.isCamera,
        );
      }
      isLeader = ref.isLeader;
      isCamera = ref.isCamera;

      session = fetched;
      _rememberSession(ref.sessionId, ref.isLeader, camera: ref.isCamera);
      _log.info('Wznowiono sesję ${ref.sessionId} (stan=${fetched.state})');
      _connectWs();

      if (fetched.hasMeasurement || fetched.isDone) {
        await _fetchMeasurement();
      }

      _setLoading(false);
      return true;
    } catch (e, st) {
      _log.warn('Nie udało się wznowić sesji ${ref.sessionId} '
          '(server=${ref.serverUrl})', e, st);
      sessionId = null;
      session = null;
      _setError('Nie można wrócić do sesji ${ref.sessionId}: $e');
      return false;
    }
  }

  /// Trwale usuwa sesję na serwerze (wraz z danymi) i z lokalnej historii.
  Future<void> deleteKnownSession(SessionRef ref) async {
    try {
      await _api.deleteSession(ref.sessionId);
      _log.info('Usunięto sesję ${ref.sessionId} na serwerze');
    } catch (e, st) {
      // Sesja mogła już nie istnieć na serwerze - i tak czyścimy lokalnie.
      _log.warn('Usuwanie sesji ${ref.sessionId} na serwerze nie powiodło się '
          '- usuwam tylko lokalnie', e, st);
    }
    _forgetSession(ref.sessionId);
    if (sessionId == ref.sessionId) {
      _ws?.sink.close();
      _wsSub?.cancel();
      _ws = null;
      _wsSub = null;
      sessionId = null;
      session = null;
      _resetTransientState();
    }
    notifyListeners();
  }

  Future<void> refreshSession() async {
    if (sessionId == null) return;
    try {
      session = await _api.getSession(sessionId!);
      notifyListeners();
    } catch (e, st) {
      _log.warn('Nie udało się odświeżyć sesji $sessionId', e, st);
    }
  }

  // -------------------------------------------------------------------------
  // WebSocket
  // -------------------------------------------------------------------------

  void _connectWs() {
    if (sessionId == null || deviceId.isEmpty) return;

    String wsBase = serverUrl;
    if (wsBase.startsWith('https://')) {
      wsBase = 'wss://${wsBase.substring(8)}';
    } else if (wsBase.startsWith('http://')) {
      wsBase = 'ws://${wsBase.substring(7)}';
    }
    final wsUrl = '$wsBase/ws/$sessionId/$deviceId';

    _ws?.sink.close();
    _wsSub?.cancel();

    try {
      _log.info('Łączenie WebSocket: $wsUrl');
      _ws = WebSocketChannel.connect(Uri.parse(wsUrl));
      _wsSub = _ws!.stream.listen(
        (raw) {
          try {
            _handleWsMsg(jsonDecode(raw as String) as Map<String, dynamic>);
          } catch (e, st) {
            _log.warn('Nie udało się sparsować wiadomości WS: $raw', e, st);
          }
        },
        onError: (Object e, StackTrace st) {
          wsConnected = false;
          _log.error('Błąd strumienia WebSocket ($wsUrl)', e, st);
          _setError('WebSocket: $e');
        },
        onDone: () {
          wsConnected = false;
          _log.warn('WebSocket rozłączony ($wsUrl), kod=${_ws?.closeCode}, '
              'powód=${_ws?.closeReason}');
          _setInfo('WebSocket rozłączony');
          notifyListeners();
          if (sessionId != null) {
            Future.delayed(const Duration(seconds: 3), () {
              if (sessionId != null && !wsConnected) {
                _log.info('Auto-reconnect WebSocket...');
                _connectWs();
              }
            });
          }
        },
      );

      // Ping startowy - pomiar offsetu czasu serwera
      Future.delayed(const Duration(milliseconds: 400), () {
        try {
          _pingSentAt = DateTime.now().millisecondsSinceEpoch / 1000.0;
          _ws?.sink.add(jsonEncode({'action': 'ping'}));
        } catch (e, st) {
          _log.warn('Nie udało się wysłać pinga startowego WS', e, st);
        }
      });
    } catch (e, st) {
      _log.error('Nie można połączyć WebSocket ($wsUrl)', e, st);
      _setError('Nie można połączyć WebSocket: $e');
    }
  }

  void reconnectWs() => _connectWs();

  void _handleWsMsg(Map<String, dynamic> msg) {
    wsLog.add(msg);
    if (wsLog.length > 30) wsLog.removeAt(0);

    final event = msg['event'] as String?;
    _log.info('WS event: ${event ?? "(brak pola event)"}');
    if (event == null) {
      _log.warn('Wiadomość WS bez pola "event": $msg');
    }

    switch (event) {
      case 'ping':
        return; // server keepalive — no state change

      case 'pong':
        final st = (msg['t'] as num?)?.toDouble() ?? 0.0;
        final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
        final rtt = _pingSentAt > 0 ? now - _pingSentAt : 0.0;
        serverTimeOffset = st - now + rtt / 2;
        wsConnected = true;
        _log.info('WS połączony, offset czasu serwera = '
            '${serverTimeOffset.toStringAsFixed(3)} s, RTT = '
            '${(rtt * 1000).toStringAsFixed(0)} ms');
        break;

      case 'session_state':
        // Backend pushes current state on WS connect — apply directly, no extra HTTP call
        final rawDevices = msg['devices'] as List?;
        if (session != null && rawDevices != null) {
          session = SessionData(
            sessionId: session!.sessionId,
            state: msg['state'] as String? ?? session!.state,
            devices: rawDevices
                .map((d) => DeviceInfo.fromJson(d as Map<String, dynamic>))
                .toList(),
            createdAt: session!.createdAt,
            hasCalibration: msg['has_calibration'] as bool? ?? session!.hasCalibration,
            hasMeasurement: msg['has_measurement'] as bool? ?? session!.hasMeasurement,
          );
        } else {
          refreshSession();
        }
        break;

      case 'device_ws_connected':
        _setInfo('Urządzenie ${msg['device_id']} połączyło WebSocket');
        refreshSession();
        break;

      case 'device_ws_disconnected':
        _setInfo('Urządzenie ${msg['device_id']} rozłączyło WebSocket');
        refreshSession();
        break;

      case 'device_joined':
        _setInfo('Urządzenie ${msg['device_id']} dołączyło do sesji');
        refreshSession();
        break;

      case 'device_left':
        _setInfo('Urządzenie ${msg['device_id']} opuściło sesję');
        refreshSession();
        break;

      case 'calibration_done':
        final err = (msg['reproj_error'] as num?)?.toStringAsFixed(3) ?? '?';
        _setInfo('Kalibracja zakończona - błąd reprojekcji: $err px');
        refreshSession();
        break;

      case 'capture_trigger':
        captureTriggerAt = (msg['at'] as num?)?.toDouble();
        notifyListeners();
        break;

      case 'calib_trigger':
        calibTriggerAt = (msg['at'] as num?)?.toDouble();
        notifyListeners();
        break;

      case 'calib_frame_uploaded':
        final upDeviceId = msg['device_id'] as String?;
        final newTotal = (msg['total_frames'] as num?)?.toInt();
        if (upDeviceId != null && newTotal != null && session != null) {
          session = session!.copyWithDevices(
            session!.devices.map((d) => d.deviceId == upDeviceId
                ? d.copyWith(calibFrameCount: newTotal)
                : d).toList(),
          );
        }
        break;

      case 'device_captured':
        _setInfo('Urządzenie ${msg['device_id']} wykonało zdjęcie');
        refreshSession();
        break;

      case 'measurement_done':
        final w = (msg['width_mm'] as num?)?.toStringAsFixed(0) ?? '?';
        final l = (msg['length_mm'] as num?)?.toStringAsFixed(0) ?? '?';
        final h = (msg['height_mm'] as num?)?.toStringAsFixed(0) ?? '?';
        final vol = (msg['volume_voxel_l'] as num?)?.toStringAsFixed(2);
        final volStr = vol == null ? '' : ', ~$vol l';
        _setInfo('Pomiar zakończony: $w × $l × $h mm$volStr');
        refreshSession();
        _fetchMeasurement();
        break;

      case 'error':
        _log.warn('Serwer zgłosił błąd przez WS: ${msg['message']}');
        _setError('Serwer: ${msg['message'] ?? 'Nieznany błąd'}');
        refreshSession();
        break;
    }

    notifyListeners();
  }

  Future<void> _fetchMeasurement() async {
    if (sessionId == null) return;
    try {
      measurement = await _api.getMeasurement(sessionId!);
      notifyListeners();
    } catch (e, st) {
      _log.warn('Nie udało się pobrać wyników pomiaru dla sesji $sessionId', e, st);
      _setError('Błąd pobierania wyników: $e');
    }
  }

  /// Broadcasts a synchronized calibration capture trigger to all devices.
  Future<void> triggerCalibCapture({int delayMs = 3000}) async {
    if (sessionId == null) return;
    try {
      await _api.triggerCalibCapture(sessionId!, delayMs);
      _log.info('Wysłano trigger kalibracyjny (delay=${delayMs}ms)');
    } catch (e, st) {
      _log.warn('Nie udało się wyzwolić triggera kalibracyjnego (sesja $sessionId)', e, st);
      _setError(e.toString());
    }
  }

  /// Uploads a single calibration image immediately (used after synchronized capture).
  Future<bool> uploadCalibImageNow(Uint8List bytes) async {
    if (!isCamera || sessionId == null || deviceId.isEmpty) return false;
    try {
      final resp = await _api.uploadCalibImage(sessionId!, deviceId, bytes);
      final newTotal = resp['total_frames'] as int?;
      if (newTotal != null && session != null) {
        session = session!.copyWithDevices(
          session!.devices
              .map((d) => d.deviceId == deviceId
                  ? d.copyWith(calibFrameCount: newTotal)
                  : d)
              .toList(),
        );
        notifyListeners();
      }
      _log.info('Przesłano klatkę kalibracyjną (device=$deviceId, total=$newTotal)');
      return true;
    } catch (e, st) {
      _log.warn('Przesyłanie klatki kalibracyjnej nie powiodło się '
          '(device=$deviceId)', e, st);
      _setError(e.toString());
      return false;
    }
  }

  Future<void> startCalibration() async {
    if (sessionId == null) return;
    _setLoading(true);
    try {
      await _api.computeCalibration(sessionId!);
      _setInfo('Kalibracja uruchomiona w tle - czekaj na wynik...');
    } catch (e, st) {
      _log.warn('Nie udało się uruchomić kalibracji dla sesji $sessionId', e, st);
      _setError(e.toString());
    }
    _setLoading(false);
  }

  // -------------------------------------------------------------------------
  // Przechwytywanie
  // -------------------------------------------------------------------------

  /// Rozgłasza trigger do wszystkich urządzeń przez WebSocket.
  Future<void> triggerCapture({int delayMs = 3000}) async {
    if (sessionId == null) return;
    try {
      await _api.triggerCapture(sessionId!, delayMs);
      _log.info('Wysłano trigger przechwycenia (delay=${delayMs}ms)');
    } catch (e, st) {
      _log.warn('Nie udało się wyzwolić przechwycenia (sesja $sessionId)', e, st);
      _setError(e.toString());
    }
  }

  Future<bool> uploadCaptureImage(Uint8List bytes) async {
    if (!isCamera || sessionId == null || deviceId.isEmpty) return false;
    try {
      await _api.uploadCaptureImage(sessionId!, deviceId, bytes);
      // Powiadom serwer (inne urządzenia) - at w czasie serwera
      try {
        _ws?.sink.add(jsonEncode({
          'action': 'captured',
          'at':
              DateTime.now().millisecondsSinceEpoch / 1000.0 + serverTimeOffset,
        }));
      } catch (e, st) {
        _log.warn('Zdjęcie przesłane, ale powiadomienie WS "captured" '
            'nie zostało wysłane', e, st);
      }
      await refreshSession();
      _log.info('Przesłano zdjęcie pomiarowe (${bytes.length} B, device=$deviceId)');
      return true;
    } catch (e, st) {
      _log.warn('Przesyłanie zdjęcia pomiarowego nie powiodło się '
          '(device=$deviceId, ${bytes.length} B)', e, st);
      _setError(e.toString());
      return false;
    }
  }

  Future<void> startMeasurement() async {
    if (sessionId == null) return;
    _setLoading(true);
    try {
      await _api.runMeasurement(sessionId!);
      _setInfo('Pipeline pomiaru uruchomiony - czekaj na wynik...');
    } catch (e, st) {
      _log.warn('Nie udało się uruchomić pomiaru dla sesji $sessionId', e, st);
      _setError(e.toString());
    }
    _setLoading(false);
  }

  // -------------------------------------------------------------------------
  // Test syntetyczny (bez kamer)
  // -------------------------------------------------------------------------

  Future<MeasurementResult?> runSyntheticTest() async {
    _setLoading(true);
    try {
      final result = await _api.syntheticMeasure();
      measurement = result;
      _log.info('Test syntetyczny zakończony (walidacja=${result.validationPassed})');
      _setLoading(false);
      return result;
    } catch (e, st) {
      _log.warn('Test syntetyczny nie powiódł się (server=$serverUrl)', e, st);
      _setError(e.toString());
      return null;
    }
  }

  // -------------------------------------------------------------------------
  // Pobierz wyniki (pollingowe jako backup do WS)
  // -------------------------------------------------------------------------

  Future<void> fetchMeasurementNow() async => _fetchMeasurement();

  // -------------------------------------------------------------------------
  // Rozłączenie / reset
  // -------------------------------------------------------------------------

  /// Opuszcza bieżącą sesję (wypisuje urządzenie), ale POZOSTAWIA ją w
  /// historii i na serwerze - można do niej później wrócić przez resumeSession().
  Future<void> leaveSession() async {
    _ws?.sink.close();
    _wsSub?.cancel();
    _ws = null;
    _wsSub = null;

    if (sessionId != null && deviceId.isNotEmpty) {
      try {
        // Wypisuje to urządzenie. Backend zachowuje sesję i jej dane.
        await _api.leaveDevice(sessionId!, deviceId);
        _log.info('Opuszczono sesję $sessionId (device=$deviceId)');
      } catch (e, st) {
        _log.warn('Wypisanie urządzenia z sesji $sessionId nie powiodło się '
            '- czyszczę stan lokalny mimo to', e, st);
      }
    }

    sessionId = null;
    session = null;
    _resetTransientState();
    notifyListeners();
  }

  @override
  void dispose() {
    _ws?.sink.close();
    _wsSub?.cancel();
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Generatory ID (fallback gdy brak zapisu w SharedPreferences)
  // -------------------------------------------------------------------------

  static String _generateDeviceId() {
    final r = Random();
    return 'device_${r.nextInt(9000) + 1000}';
  }

  static String _generateMac() {
    final r = Random();
    String hex(int v) => v.toRadixString(16).padLeft(2, '0').toUpperCase();
    return '${hex(r.nextInt(256))}:${hex(r.nextInt(256))}:${hex(r.nextInt(256))}'
        ':${hex(r.nextInt(256))}:${hex(r.nextInt(256))}:${hex(r.nextInt(256))}';
  }
}
