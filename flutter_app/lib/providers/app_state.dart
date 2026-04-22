// app_state.dart — główny ChangeNotifier zarządzający stanem aplikacji
// Obsługuje: sesję, WebSocket, kalibrację, przechwytywanie, pomiar.

import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/models.dart';
import '../services/api_service.dart';

class AppState extends ChangeNotifier {
  // -------------------------------------------------------------------------
  // Konfiguracja serwera
  // -------------------------------------------------------------------------

  String serverUrl = 'http://192.168.1.1:8000';

  // -------------------------------------------------------------------------
  // Tożsamość urządzenia
  // -------------------------------------------------------------------------

  String deviceId = _generateDeviceId();
  String mac = _generateMac();
  bool isLeader = true;

  // -------------------------------------------------------------------------
  // Stan sesji
  // -------------------------------------------------------------------------

  String? sessionId;
  SessionData? session;
  MeasurementResult? measurement;

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

  /// true gdy WebSocket jest aktywny i odebrał co najmniej jeden pong.
  bool wsConnected = false;

  /// Timestamp (Unix sek.) kiedy urządzenie powinno zrobić zdjęcie.
  double? captureTriggerAt;

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
    error = msg;
    isLoading = false;
    notifyListeners();
  }

  void _setInfo(String msg) {
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
    // Nie wywołujemy notifyListeners — caller zrobi to sam po starcie countdown.
  }

  // -------------------------------------------------------------------------
  // Połączenie / health
  // -------------------------------------------------------------------------

  Future<bool> testConnection() async {
    return ApiService(serverUrl).healthCheck();
  }

  // -------------------------------------------------------------------------
  // Tworzenie i dołączanie do sesji
  // -------------------------------------------------------------------------

  /// Tworzy nową sesję i dołącza jako lider lub follower.
  Future<bool> createAndJoin(String did, String m, bool leader) async {
    _setLoading(true);
    try {
      serverUrl = serverUrl.trim();
      deviceId = did;
      mac = m;
      isLeader = leader;

      final sess = await _api.createSession();
      sessionId = sess.sessionId;

      final joined = await _api.joinSession(sess.sessionId, did, m, leader);
      session = joined;

      _connectWs();
      _setLoading(false);
      return true;
    } catch (e) {
      sessionId = null;
      _setError(e.toString());
      return false;
    }
  }

  /// Dołącza do istniejącej sesji (follower).
  Future<bool> joinExisting(String sid, String did, String m, bool leader) async {
    _setLoading(true);
    try {
      serverUrl = serverUrl.trim();
      deviceId = did;
      mac = m;
      isLeader = leader;
      sessionId = sid;

      final joined = await _api.joinSession(sid, did, m, leader);
      session = joined;

      _connectWs();
      _setLoading(false);
      return true;
    } catch (e) {
      sessionId = null;
      _setError(e.toString());
      return false;
    }
  }

  Future<void> refreshSession() async {
    if (sessionId == null) return;
    try {
      session = await _api.getSession(sessionId!);
      notifyListeners();
    } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // WebSocket
  // -------------------------------------------------------------------------

  void _connectWs() {
    if (sessionId == null || deviceId.isEmpty) return;

    final wsBase = serverUrl
        .replaceFirst(RegExp(r'^https://'), 'wss://')
        .replaceFirst(RegExp(r'^http://'), 'ws://');
    final wsUrl = '$wsBase/ws/$sessionId/$deviceId';

    _ws?.sink.close();
    _wsSub?.cancel();

    try {
      _ws = WebSocketChannel.connect(Uri.parse(wsUrl));
      _wsSub = _ws!.stream.listen(
        (raw) {
          try {
            _handleWsMsg(jsonDecode(raw as String) as Map<String, dynamic>);
          } catch (_) {}
        },
        onError: (Object e) {
          wsConnected = false;
          _setError('WebSocket: $e');
        },
        onDone: () {
          wsConnected = false;
          _setInfo('WebSocket rozłączony');
          notifyListeners();
        },
      );

      // Ping startowy — pomiar offsetu czasu serwera
      Future.delayed(const Duration(milliseconds: 400), () {
        _ws?.sink.add(jsonEncode({'action': 'ping'}));
      });
    } catch (e) {
      _setError('Nie można połączyć WebSocket: $e');
    }
  }

  void reconnectWs() => _connectWs();

  void _handleWsMsg(Map<String, dynamic> msg) {
    wsLog.add(msg);
    if (wsLog.length > 30) wsLog.removeAt(0);

    final event = msg['event'] as String?;

    switch (event) {
      case 'pong':
        final st = (msg['t'] as num?)?.toDouble() ?? 0.0;
        serverTimeOffset =
            st - DateTime.now().millisecondsSinceEpoch / 1000.0;
        wsConnected = true;
        break;

      case 'device_joined':
        _setInfo('📱 ${msg['device_id']} dołączył do sesji');
        refreshSession();
        break;

      case 'calibration_done':
        final err = (msg['reproj_error'] as num?)?.toStringAsFixed(3) ?? '?';
        _setInfo('✅ Kalibracja zakończona! Błąd reprojekcji: $err px');
        refreshSession();
        break;

      case 'capture_trigger':
        captureTriggerAt = (msg['at'] as num?)?.toDouble();
        notifyListeners(); // Ekran Capture nasłuchuje tej zmiany
        break;

      case 'device_captured':
        _setInfo('📸 ${msg['device_id']} wykonał zdjęcie');
        refreshSession();
        break;

      case 'measurement_done':
        final w = (msg['width_mm'] as num?)?.toStringAsFixed(0) ?? '?';
        final l = (msg['length_mm'] as num?)?.toStringAsFixed(0) ?? '?';
        final h = (msg['height_mm'] as num?)?.toStringAsFixed(0) ?? '?';
        _setInfo('📐 Pomiar gotowy: $w × $l × $h mm');
        refreshSession();
        _fetchMeasurement();
        break;

      case 'error':
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
    } catch (e) {
      _setError('Błąd pobierania wyników: $e');
    }
  }

  // -------------------------------------------------------------------------
  // Kalibracja
  // -------------------------------------------------------------------------

  Future<bool> uploadCalibImage(Uint8List bytes) async {
    if (sessionId == null || deviceId.isEmpty) return false;
    try {
      await _api.uploadCalibImage(sessionId!, deviceId, bytes);
      await refreshSession();
      return true;
    } catch (e) {
      _setError(e.toString());
      return false;
    }
  }

  Future<void> startCalibration() async {
    if (sessionId == null) return;
    _setLoading(true);
    try {
      await _api.computeCalibration(sessionId!);
      _setInfo('Kalibracja uruchomiona w tle — czekaj na wynik...');
    } catch (e) {
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
    } catch (e) {
      _setError(e.toString());
    }
  }

  Future<bool> uploadCaptureImage(Uint8List bytes) async {
    if (sessionId == null || deviceId.isEmpty) return false;
    try {
      await _api.uploadCaptureImage(sessionId!, deviceId, bytes);
      // Powiadom serwer (inne urządzenia) — at w czasie serwera
      _ws?.sink.add(jsonEncode({
        'action': 'captured',
        'at': DateTime.now().millisecondsSinceEpoch / 1000.0 + serverTimeOffset,
      }));
      await refreshSession();
      return true;
    } catch (e) {
      _setError(e.toString());
      return false;
    }
  }

  Future<void> startMeasurement() async {
    if (sessionId == null) return;
    _setLoading(true);
    try {
      await _api.runMeasurement(sessionId!);
      _setInfo('Pipeline pomiaru uruchomiony — czekaj na wynik...');
    } catch (e) {
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
      _setLoading(false);
      return result;
    } catch (e) {
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

  Future<void> leaveSession() async {
    _ws?.sink.close();
    _wsSub?.cancel();
    _ws = null;
    _wsSub = null;

    if (sessionId != null) {
      try {
        await _api.deleteSession(sessionId!);
      } catch (_) {}
    }

    sessionId = null;
    session = null;
    measurement = null;
    captureTriggerAt = null;
    wsConnected = false;
    serverTimeOffset = 0.0;
    wsLog.clear();
    notifyListeners();
  }

  // -------------------------------------------------------------------------
  // Generatory ID (przy braku SharedPreferences)
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
