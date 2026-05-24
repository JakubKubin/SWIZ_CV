// api_service.dart - klient HTTP do backendu FastAPI

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/models.dart'; // includes FrameInfo, SessionData, MeasurementResult
import '../utils/log.dart';

class ApiService {
  static const _log = Log('ApiService');

  final String baseUrl;

  const ApiService(this.baseUrl);

  // -------------------------------------------------------------------------
  // Pomocnicy
  // -------------------------------------------------------------------------

  /// Nagłówki bazowe dla każdego żądania.
  /// ngrok-skip-browser-warning pomija stronę ostrzeżenia ngrok free tier
  /// (wymagane przy publicznym tunelu ngrok; nieszkodliwe dla innych URL-i).
  static const Map<String, String> _baseHeaders = {
    'ngrok-skip-browser-warning': 'true',
  };

  static Map<String, String> _jsonHeaders() => {
        ..._baseHeaders,
        'Content-Type': 'application/json',
      };

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
          .get(_uri('/health'), headers: _baseHeaders)
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
    final r = await http.post(_uri('/sessions'), headers: _jsonHeaders());
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> joinSession(
    String sid,
    String deviceId,
    String mac,
    bool isLeader, {
    bool isCamera = true,
  }) async {
    final r = await http.post(
      _uri('/sessions/$sid/join'),
      headers: _jsonHeaders(),
      body: jsonEncode({
        'device_id': deviceId,
        'mac': mac,
        'is_leader': isLeader,
        'is_camera': isCamera,
      }),
    );
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> getSession(String sid) async {
    final r = await http.get(_uri('/sessions/$sid'), headers: _baseHeaders);
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<void> deleteSession(String sid) async {
    final r = await http.delete(_uri('/sessions/$sid'), headers: _baseHeaders);
    if (r.statusCode != 204 && r.statusCode != 404) await _checkStatus(r);
  }

  /// Usuwa jedno urządzenie z sesji. Backend automatycznie usuwa sesję
  /// gdy nie zostanie żadne urządzenie.
  Future<void> leaveDevice(String sid, String deviceId) async {
    final r = await http.delete(
      _uri('/sessions/$sid/devices/$deviceId'),
      headers: _baseHeaders,
    );
    if (r.statusCode != 204 && r.statusCode != 404) await _checkStatus(r);
  }

  /// Leader removes another device from the session.
  Future<void> removeDevice(
    String sid,
    String targetDeviceId,
    String requesterId,
  ) async {
    final r = await http.delete(
      _uri('/sessions/$sid/devices/$targetDeviceId'
          '?requester_id=${Uri.encodeComponent(requesterId)}'),
      headers: _baseHeaders,
    );
    if (r.statusCode != 204 && r.statusCode != 404) await _checkStatus(r);
  }

  Future<SessionData> promoteDevice(
    String sid,
    String targetDeviceId,
    String requesterId,
  ) async {
    final r = await http.post(
      _uri('/sessions/$sid/devices/$targetDeviceId/promote'
          '?requester_id=${Uri.encodeComponent(requesterId)}'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> patchDevice(
    String sid,
    String targetDeviceId,
    String requesterId, {
    required bool isCamera,
  }) async {
    final r = await http.patch(
      _uri('/sessions/$sid/devices/$targetDeviceId'
          '?requester_id=${Uri.encodeComponent(requesterId)}'),
      headers: _jsonHeaders(),
      body: jsonEncode({'is_camera': isCamera}),
    );
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<List<SessionData>> listSessions() async {
    final r = await http.get(_uri('/sessions'), headers: _baseHeaders);
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
    req.headers.addAll(_baseHeaders);
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
    final r = await http.post(
      _uri('/sessions/$sid/calibration/compute'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
  }

  Future<Map<String, dynamic>> getCalibrationStatus(String sid) async {
    final r = await http.get(
      _uri('/sessions/$sid/calibration'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  // -------------------------------------------------------------------------
  // Przechwytywanie
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> triggerCalibCapture(String sid, int delayMs) async {
    final r = await http.post(
      _uri('/sessions/$sid/calibration/trigger'),
      headers: _jsonHeaders(),
      body: jsonEncode({'delay_ms': delayMs}),
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> triggerCapture(String sid, int delayMs) async {
    final r = await http.post(
      _uri('/sessions/$sid/capture/trigger'),
      headers: _jsonHeaders(),
      body: jsonEncode({'delay_ms': delayMs}),
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<List<FrameInfo>> listCalibImages(String sid, String deviceId) async {
    final r = await http.get(
      _uri('/sessions/$sid/calibration/images/$deviceId'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return (jsonDecode(r.body) as List)
        .map((j) => FrameInfo.fromJson(j as Map<String, dynamic>))
        .toList();
  }

  Future<List<FrameInfo>> listCaptureImages(String sid, String deviceId) async {
    final r = await http.get(
      _uri('/sessions/$sid/capture/images/$deviceId'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return (jsonDecode(r.body) as List)
        .map((j) => FrameInfo.fromJson(j as Map<String, dynamic>))
        .toList();
  }

  /// Fetches raw image bytes from an absolute URL, sending the base headers.
  Future<Uint8List> getImageBytes(String url) async {
    final r = await http.get(Uri.parse(url), headers: _baseHeaders);
    if (r.statusCode >= 400) throw ApiException(r.statusCode, 'Image load failed');
    return r.bodyBytes;
  }

  Future<Map<String, dynamic>> deleteCalibPair(
    String sid,
    int frameIndex,
    String requesterId,
  ) async {
    final r = await http.delete(
      _uri('/sessions/$sid/calibration/pairs/$frameIndex'
          '?requester_id=${Uri.encodeComponent(requesterId)}'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> deleteCaptureFrame(
    String sid,
    String deviceId,
    int frameIndex,
    String requesterId,
  ) async {
    final r = await http.delete(
      _uri('/sessions/$sid/capture/images/$deviceId/$frameIndex'
          '?requester_id=${Uri.encodeComponent(requesterId)}'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> deleteCaptureImages(
    String sid,
    String targetDeviceId,
    String requesterId,
  ) async {
    final r = await http.delete(
      _uri('/sessions/$sid/capture/images/$targetDeviceId'
          '?requester_id=${Uri.encodeComponent(requesterId)}'),
      headers: _baseHeaders,
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
    req.headers.addAll(_baseHeaders);
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
    final r = await http.post(
      _uri('/sessions/$sid/measure'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
  }

  Future<MeasurementResult> getMeasurement(String sid) async {
    final r = await http.get(
      _uri('/sessions/$sid/measurement'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return MeasurementResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<String> getMeasurementReport(String sid) async {
    final r = await http.get(
      _uri('/sessions/$sid/measurement/report'),
      headers: _baseHeaders,
    );
    await _checkStatus(r);
    return r.body;
  }

  Future<MeasurementResult> syntheticMeasure() async {
    final r = await http
        .post(_uri('/measure/synthetic'), headers: _baseHeaders)
        .timeout(const Duration(seconds: 30));
    await _checkStatus(r);
    return MeasurementResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }
}
