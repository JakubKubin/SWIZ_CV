// api_service.dart - klient HTTP do backendu FastAPI

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/models.dart';
import '../utils/log.dart';

class ApiService {
  static const _log = Log('ApiService');

  final String baseUrl;

  const ApiService(this.baseUrl);

  // -------------------------------------------------------------------------
  // Pomocnicy
  // -------------------------------------------------------------------------

  Future<void> _checkStatus(http.Response r) async {
    if (r.statusCode >= 400) {
      String detail;
      try {
        detail = jsonDecode(r.body)['detail'] as String? ?? r.body;
      } catch (_) {
        detail = r.body;
      }
      final req = r.request;
      _log.warn('HTTP ${r.statusCode} ${req?.method} ${req?.url}: $detail');
      throw ApiException(r.statusCode, detail);
    }
  }

  /// Obsługa błędów dla odpowiedzi multipart (streamed) - loguje i rzuca.
  Never _failMultipart(int statusCode, String method, Uri url, String body) {
    String detail;
    try {
      detail = jsonDecode(body)['detail'] as String? ?? body;
    } catch (_) {
      detail = body;
    }
    _log.warn('HTTP $statusCode $method $url: $detail');
    throw ApiException(statusCode, detail);
  }

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  // -------------------------------------------------------------------------
  // Health
  // -------------------------------------------------------------------------

  Future<bool> healthCheck() async {
    try {
      final r = await http
          .get(_uri('/health'))
          .timeout(const Duration(seconds: 5));
      if (r.statusCode != 200) {
        _log.warn('Health check zwrócił HTTP ${r.statusCode} ($baseUrl)');
      }
      return r.statusCode == 200;
    } catch (e, st) {
      _log.warn('Health check nie powiódł się ($baseUrl)', e, st);
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Sesje
  // -------------------------------------------------------------------------

  Future<SessionData> createSession() async {
    final r = await http.post(_uri('/sessions'),
        headers: {'Content-Type': 'application/json'});
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> joinSession(
    String sid,
    String deviceId,
    String mac,
    bool isLeader,
  ) async {
    final r = await http.post(
      _uri('/sessions/$sid/join'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'device_id': deviceId, 'mac': mac, 'is_leader': isLeader}),
    );
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> getSession(String sid) async {
    final r = await http.get(_uri('/sessions/$sid'));
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<void> deleteSession(String sid) async {
    final r = await http.delete(_uri('/sessions/$sid'));
    if (r.statusCode != 204 && r.statusCode != 404) await _checkStatus(r);
  }

  /// Usuwa jedno urządzenie z sesji. Backend automatycznie usuwa sesję
  /// gdy nie zostanie żadne urządzenie.
  Future<void> leaveDevice(String sid, String deviceId) async {
    final r = await http.delete(_uri('/sessions/$sid/devices/$deviceId'));
    if (r.statusCode != 204 && r.statusCode != 404) await _checkStatus(r);
  }

  Future<List<SessionData>> listSessions() async {
    final r = await http.get(_uri('/sessions'));
    await _checkStatus(r);
    return (jsonDecode(r.body) as List)
        .map((j) => SessionData.fromJson(j as Map<String, dynamic>))
        .toList();
  }

  // -------------------------------------------------------------------------
  // Kalibracja
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> uploadCalibImage(
    String sid,
    String deviceId,
    Uint8List bytes,
  ) async {
    final url = _uri('/sessions/$sid/calibration/images');
    final req = http.MultipartRequest('POST', url);
    req.fields['device_id'] = deviceId;
    req.files.add(http.MultipartFile.fromBytes('file', bytes, filename: 'frame.jpg'));

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode >= 400) {
      _failMultipart(streamed.statusCode, 'POST', url, body);
    }
    return jsonDecode(body) as Map<String, dynamic>;
  }

  Future<void> computeCalibration(String sid) async {
    final r = await http.post(_uri('/sessions/$sid/calibration/compute'));
    await _checkStatus(r);
  }

  Future<Map<String, dynamic>> getCalibrationStatus(String sid) async {
    final r = await http.get(_uri('/sessions/$sid/calibration'));
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  // -------------------------------------------------------------------------
  // Przechwytywanie
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> triggerCapture(String sid, int delayMs) async {
    final r = await http.post(
      _uri('/sessions/$sid/capture/trigger'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'delay_ms': delayMs}),
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> uploadCaptureImage(
    String sid,
    String deviceId,
    Uint8List bytes,
  ) async {
    final url = _uri('/sessions/$sid/capture/images');
    final req = http.MultipartRequest('POST', url);
    req.fields['device_id'] = deviceId;
    req.files.add(http.MultipartFile.fromBytes('file', bytes, filename: 'capture.jpg'));

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode >= 400) {
      _failMultipart(streamed.statusCode, 'POST', url, body);
    }
    return jsonDecode(body) as Map<String, dynamic>;
  }

  // -------------------------------------------------------------------------
  // Pomiar
  // -------------------------------------------------------------------------

  Future<void> runMeasurement(String sid) async {
    final r = await http.post(_uri('/sessions/$sid/measure'));
    await _checkStatus(r);
  }

  Future<MeasurementResult> getMeasurement(String sid) async {
    final r = await http.get(_uri('/sessions/$sid/measurement'));
    await _checkStatus(r);
    return MeasurementResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<String> getMeasurementReport(String sid) async {
    final r = await http.get(_uri('/sessions/$sid/measurement/report'));
    await _checkStatus(r);
    return r.body;
  }

  Future<MeasurementResult> syntheticMeasure() async {
    final r = await http
        .post(_uri('/measure/synthetic'))
        .timeout(const Duration(seconds: 30));
    await _checkStatus(r);
    return MeasurementResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }
}
